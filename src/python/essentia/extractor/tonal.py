# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

import numpy
import essentia
import sys
from essentia import EssentiaError, INFO
from essentia.progress import Progress


namespace = 'tonal'
dependencies = None


def normalize(hpcp):
    m = max(hpcp)
    for i in range(len(hpcp)):
        hpcp[i] = hpcp[i] / m
    return hpcp


def compute(audio, pool, options):

    INFO('Computing Tonal descriptors...')

    sampleRate  = options['sampleRate']
    frameSize   = options['frameSize']
    hopSize     = options['hopSize']
    zeroPadding = options['zeroPadding']
    windowType  = options['windowType']

    frames = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window = essentia.Windowing(size = frameSize, zeroPadding = zeroPadding, type = windowType)
    spectrum = essentia.Spectrum(size = (frameSize + zeroPadding) / 2)
    spectral_peaks = essentia.SpectralPeaks(maxPeaks = 10000, magnitudeThreshold = 0.00001, minFrequency = 40, maxFrequency = 5000, orderBy = "frequency")
    tuning = essentia.TuningFrequency()

    # computing the tuning frequency
    tuning_frequency = 440.0

    for frame in frames:

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)

        (frame_frequencies, frame_magnitudes) = spectral_peaks(frame_spectrum)

        #if len(frame_frequencies) > 0:
        (tuning_frequency, tuning_cents) = tuning(frame_frequencies, frame_magnitudes)

    pool.add(namespace + '.' + 'tuning_frequency', tuning_frequency)#, pool.GlobalScope)

    # computing the HPCPs
    spectral_whitening = essentia.SpectralWhitening()

    hpcp_key_size = 36
    hpcp_chord_size = 36
    hpcp_tuning_size = 120

    hpcp_key = essentia.HPCP(size = hpcp_key_size,
                             referenceFrequency = tuning_frequency,
                             bandPreset = False,
                             minFrequency = 40.0,
                             maxFrequency = 5000.0,
                             weightType = 'squaredCosine',
                             nonLinear = False,
                             windowSize = 4.0/3.0,
                             sampleRate = sampleRate)

    hpcp_chord = essentia.HPCP(size = hpcp_chord_size,
                               referenceFrequency = tuning_frequency,
                               harmonics = 8,
                               bandPreset = True,
                               minFrequency = 40.0,
                               maxFrequency = 5000.0,
                               splitFrequency = 500.0,
                               weightType = 'cosine',
                               nonLinear = True,
                               windowSize = 0.5,
                               sampleRate = sampleRate)

    hpcp_tuning = essentia.HPCP(size = hpcp_tuning_size,
                                referenceFrequency = tuning_frequency,
                                harmonics = 8,
                                bandPreset = True,
                                minFrequency = 40.0,
                                maxFrequency = 5000.0,
                                splitFrequency = 500.0,
                                weightType = 'cosine',
                                nonLinear = True,
                                windowSize = 0.5,
                                sampleRate = sampleRate)

    # intializing the HPCP arrays
    hpcps_key = []
    hpcps_chord = []
    hpcps_tuning = []

    # computing HPCP loop
    frames = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)

    total_frames = frames.num_frames()
    n_frames = 0
    start_of_frame = -frameSize * 0.5

    progress = Progress(total = total_frames)


    for frame in frames:

        #frameScope = [ start_of_frame / sampleRate, (start_of_frame + frameSize) / sampleRate ]
        #pool.setCurrentScope(frameScope)

        if options['skipSilence'] and essentia.isSilent(frame):
          total_frames -= 1
          start_of_frame += hopSize
          continue

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)

        # spectral peaks
        (frame_frequencies, frame_magnitudes) = spectral_peaks(frame_spectrum)

        if (len(frame_frequencies) > 0):
           # spectral_whitening
           frame_magnitudes_white = spectral_whitening(frame_spectrum, frame_frequencies, frame_magnitudes)
           frame_hpcp_key = hpcp_key(frame_frequencies, frame_magnitudes_white)
           frame_hpcp_chord = hpcp_chord(frame_frequencies, frame_magnitudes_white)
           frame_hpcp_tuning = hpcp_tuning(frame_frequencies, frame_magnitudes_white)
        else:
           frame_hpcp_key = essentia.array([0] * hpcp_key_size)
           frame_hpcp_chord = essentia.array([0] * hpcp_chord_size)
           frame_hpcp_tuning = essentia.array([0] * hpcp_tuning_size)

        # key HPCP
        hpcps_key.append(frame_hpcp_key)

        # add HPCP to the pool
        pool.add(namespace + '.' +'hpcp', frame_hpcp_key)

        # chords HPCP
        hpcps_chord.append(frame_hpcp_chord)

        # tuning system HPCP
        hpcps_tuning.append(frame_hpcp_tuning)

        # display of progress report
        progress.update(n_frames)

        n_frames += 1
        start_of_frame += hopSize

    progress.finish()

    # check if silent file
    if len(hpcps_key) == 0:
       raise EssentiaError('This is a silent file!')

    # key detection
    key_detector = essentia.Key(profileType = 'temperley')
    average_hpcps_key = numpy.average(essentia.array(hpcps_key), axis=0)
    average_hpcps_key = normalize(average_hpcps_key)

    # thpcps
    max_arg = numpy.argmax( average_hpcps_key )
    thpcp=[]
    for i in range( max_arg, len(average_hpcps_key) ):
        thpcp.append( float(average_hpcps_key[i]) )
    for i in range( max_arg ):
        thpcp.append( float(average_hpcps_key[i]) )
    pool.add(namespace + '.' +'thpcp', thpcp)#, pool.GlobalScope  )

    (key, scale, key_strength, first_to_second_relative_strength) = key_detector(essentia.array(average_hpcps_key))
    pool.add(namespace + '.' +'key_key', key)#, pool.GlobalScope)
    pool.add(namespace + '.' +'key_scale', scale)#, pool.GlobalScope)
    pool.add(namespace + '.' +'key_strength', key_strength)#, pool.GlobalScope)

    # chord detection
    chord_detector = essentia.Key(profileType = 'tonictriad', usePolyphony = False)
    hpcp_frameSize = 2.0 # 2 seconds
    hpcp_number = int(hpcp_frameSize * (sampleRate / hopSize - 1))

    for hpcp_index in range(len(hpcps_chord)):

        hpcp_index_begin = max(0, hpcp_index - hpcp_number)
        hpcp_index_end = min(hpcp_index + hpcp_number, len(hpcps_chord))
        average_hpcps_chord = numpy.average(essentia.array(hpcps_chord[hpcp_index_begin : hpcp_index_end]), axis=0)
        average_hpcps_chord = normalize(average_hpcps_chord)
        (key, scale, strength, first_to_second_relative_strength) = chord_detector(essentia.array(average_hpcps_chord))

        if scale == 'minor':
           chord = key + 'm'
        else:
           chord = key

        frame_second_scope = [hpcp_index_begin * hopSize / sampleRate, hpcp_index_end * hopSize / sampleRate]
        pool.add(namespace + '.' +'chords_progression', chord)#, frame_second_scope)
        pool.add(namespace + '.' +'chords_strength', strength)#, frame_second_scope)

    # tuning system features
    keydetector	= essentia.Key(profileType = 'diatonic')
    average_hpcps_tuning = numpy.average(essentia.array(hpcps_tuning), axis=0)
    average_hpcps_tuning = normalize(average_hpcps_tuning)
    (key, scale, diatonic_strength, first_to_second_relative_strength) = keydetector(essentia.array(average_hpcps_tuning))

    pool.add(namespace + '.' +'tuning_diatonic_strength', diatonic_strength)#, pool.GlobalScope)

    (equal_tempered_deviation,
     nontempered_energy_ratio,
     nontempered_peaks_energy_ratio) = essentia.HighResolutionFeatures()(average_hpcps_tuning)

    pool.add(namespace + '.' +'tuning_equal_tempered_deviation', equal_tempered_deviation)#, pool.GlobalScope)
    pool.add(namespace + '.' +'tuning_nontempered_energy_ratio', nontempered_energy_ratio)#, pool.GlobalScope)
    pool.add(namespace + '.' +'tuning_nontempered_peaks_energy_ratio', nontempered_peaks_energy_ratio)#, pool.GlobalScope)
