# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

import essentia
from essentia.progress import Progress
# convenience function to get descriptor names from the pool without being
# prepended by the namespace
from essentia.essentia_extractor import descriptorNames
import numpy
import sys
from essentia import INFO

namespace = 'lowlevel'
dependencies = None


def is_silent_threshold(frame, silence_threshold_dB):
    p = essentia.instantPower( frame )
    silence_threshold = pow(10.0, (silence_threshold_dB / 10.0))
    if p < silence_threshold:
       return 1.0
    else:
       return 0.0

def spectralContrastPCA(scPool, pool):
    scCoeffs = scPool.value('lowlevel.sccoeffs')
    scValleys = scPool.value('lowlevel.scvalleys')
    frames = len(scCoeffs)
    coeffs = len(scCoeffs[0])
    #merged = numpy.zeros([frames, coeffs], dtype='f4')
    merged = numpy.zeros(2*coeffs, dtype='f4')
    for i in range(frames):
        k = 0
        for j in range(coeffs):
            merged[k] = scCoeffs[i][j]
            merged[k+1] = scValleys[i][j]
            k+= 2
        scPool.add('contrast', merged)
    pca = essentia.PCA(namespaceIn = 'contrast', namespaceOut = 'contrast')(scPool)
    pool.add(namespace + '.' + 'spectral_contrast', pca.value('contrast'))

def compute(audio, pool, options):

    # analysis parameters
    sampleRate = options['sampleRate']
    frameSize  = options['frameSize']
    hopSize    = options['hopSize']
    windowType = options['windowType']

    # temporal descriptors
    lpc = essentia.LPC(order = 10, type = 'warped', sampleRate = sampleRate)
    zerocrossingrate = essentia.ZeroCrossingRate()

    # frame algorithms
    frames = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window = essentia.Windowing(size = frameSize, zeroPadding = 0, type = windowType)
    spectrum = essentia.Spectrum(size = frameSize)

    # spectral algorithms
    barkbands = essentia.BarkBands(sampleRate = sampleRate)
    centralmoments = essentia.SpectralCentralMoments()
    crest = essentia.Crest()
    centroid = essentia.SpectralCentroid()
    decrease = essentia.SpectralDecrease()
    spectral_contrast = essentia.SpectralContrast(frameSize = frameSize,
                                                  sampleRate = sampleRate,
                                                  numberBands = 6,
                                                  lowFrequencyBound = 20,
                                                  highFrequencyBound = 11000,
                                                  neighbourRatio = 0.4,
                                                  staticDistribution = 0.15)
    distributionshape = essentia.DistributionShape()
    energy = essentia.Energy()
    # energyband_bass, energyband_middle and energyband_high parameters come from "standard" hi-fi equalizers
    energyband_bass = essentia.EnergyBand(startCutoffFrequency = 20.0, stopCutoffFrequency = 150.0, sampleRate = sampleRate)
    energyband_middle_low = essentia.EnergyBand(startCutoffFrequency = 150.0, stopCutoffFrequency = 800.0, sampleRate = sampleRate)
    energyband_middle_high = essentia.EnergyBand(startCutoffFrequency = 800.0, stopCutoffFrequency = 4000.0, sampleRate = sampleRate)
    energyband_high = essentia.EnergyBand(startCutoffFrequency = 4000.0, stopCutoffFrequency = 20000.0, sampleRate = sampleRate)
    flatnessdb = essentia.FlatnessDB()
    flux = essentia.Flux()
    harmonic_peaks = essentia.HarmonicPeaks()
    hfc = essentia.HFC()
    mfcc = essentia.MFCC()
    rolloff = essentia.RollOff()
    rms = essentia.RMS()
    strongpeak = essentia.StrongPeak()

    # pitch algorithms
    pitch_detection = essentia.PitchDetection(frameSize = frameSize, sampleRate = sampleRate)
    pitch_salience = essentia.PitchSalience()

    # dissonance
    spectral_peaks = essentia.SpectralPeaks(sampleRate = sampleRate, orderBy='frequency')
    dissonance = essentia.Dissonance()

    # spectral complexity
    # magnitudeThreshold = 0.005 is hardcoded for a "blackmanharris62" frame
    spectral_complexity = essentia.SpectralComplexity(magnitudeThreshold = 0.005)

    INFO('Computing Low-Level descriptors...')

    # used for a nice progress display
    total_frames = frames.num_frames()
    n_frames = 0
    start_of_frame = -frameSize*0.5

    pitches, pitch_confidences =  [],[]

    progress = Progress(total = total_frames)

    scPool = essentia.Pool() # pool for spectral contrast

    for frame in frames:

        frameScope = [ start_of_frame / sampleRate, (start_of_frame + frameSize) / sampleRate ]
        #pool.setCurrentScope(frameScope)

        # silence rate
        pool.add(namespace + '.' + 'silence_rate_60dB', essentia.isSilent(frame))
        pool.add(namespace + '.' + 'silence_rate_30dB', is_silent_threshold(frame, -30))
        pool.add(namespace + '.' + 'silence_rate_20dB', is_silent_threshold(frame, -20))

        if options['skipSilence'] and essentia.isSilent(frame):
          total_frames -= 1
          start_of_frame += hopSize
          continue

        # temporal descriptors
        pool.add(namespace + '.' + 'zerocrossingrate', zerocrossingrate(frame))
        (frame_lpc, frame_lpc_reflection) = lpc(frame)
        pool.add(namespace + '.' + 'temporal_lpc', frame_lpc)

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)

        # spectrum-based descriptors
        power_spectrum = frame_spectrum ** 2
        pool.add(namespace + '.' + 'spectral_centroid', centroid(power_spectrum))
        pool.add(namespace + '.' + 'spectral_decrease', decrease(power_spectrum))
        pool.add(namespace + '.' + 'spectral_energy', energy(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_low', energyband_bass(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_middle_low', energyband_middle_low(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_middle_high', energyband_middle_high(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_high', energyband_high(frame_spectrum))
        pool.add(namespace + '.' + 'hfc', hfc(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_rms', rms(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_flux', flux(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_rolloff', rolloff(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_strongpeak', strongpeak(frame_spectrum))

	# central moments descriptors
	frame_centralmoments = centralmoments(power_spectrum)
        (frame_spread, frame_skewness, frame_kurtosis) = distributionshape(frame_centralmoments)
        pool.add(namespace + '.' + 'spectral_kurtosis', frame_kurtosis)
	pool.add(namespace + '.' + 'spectral_spread', frame_spread)
        pool.add(namespace + '.' + 'spectral_skewness', frame_skewness)

	# dissonance
        (frame_frequencies, frame_magnitudes) = spectral_peaks(frame_spectrum)
        frame_dissonance = dissonance(frame_frequencies, frame_magnitudes)
        pool.add(namespace + '.' + 'dissonance', frame_dissonance)

        # mfcc
        (frame_melbands, frame_mfcc) = mfcc(frame_spectrum)
        pool.add(namespace + '.' + 'mfcc', frame_mfcc)

        # spectral contrast
        (sc_coeffs, sc_valleys) = spectral_contrast(frame_spectrum)
        scPool.add(namespace + '.' + 'sccoeffs', sc_coeffs)
        scPool.add(namespace + '.' + 'scvalleys', sc_valleys)

        # barkbands-based descriptors
        frame_barkbands = barkbands(frame_spectrum)
        pool.add(namespace + '.' + 'barkbands', frame_barkbands)
        pool.add(namespace + '.' + 'spectral_crest', crest(frame_barkbands))
        pool.add(namespace + '.' + 'spectral_flatness_db', flatnessdb(frame_barkbands))
        barkbands_centralmoments = essentia.CentralMoments(range = len(frame_barkbands) - 1)
        (barkbands_spread, barkbands_skewness, barkbands_kurtosis) = distributionshape(barkbands_centralmoments(frame_barkbands))
        pool.add(namespace + '.' + 'barkbands_spread', barkbands_spread)
        pool.add(namespace + '.' + 'barkbands_skewness', barkbands_skewness)
        pool.add(namespace + '.' + 'barkbands_kurtosis', barkbands_kurtosis)

        # pitch descriptors
        frame_pitch, frame_pitch_confidence = pitch_detection(frame_spectrum)
        if frame_pitch > 0 and frame_pitch <= 20000.:
            pool.add(namespace + '.' + 'pitch', frame_pitch)
        pitches.append(frame_pitch)
        pitch_confidences.append(frame_pitch_confidence)
        pool.add(namespace + '.' + 'pitch_instantaneous_confidence', frame_pitch_confidence)

        frame_pitch_salience = pitch_salience(frame_spectrum[:-1])
        pool.add(namespace + '.' + 'pitch_salience', frame_pitch_salience)

        # spectral complexity
        pool.add(namespace + '.' + 'spectral_complexity', spectral_complexity(frame_spectrum))

        # display of progress report
        progress.update(n_frames)

        n_frames += 1
        start_of_frame += hopSize

    # if no 'temporal_zerocrossingrate' it means that this is a silent file
    if 'zerocrossingrate' not in descriptorNames(pool.descriptorNames(), namespace):
        raise essentia.EssentiaError('This is a silent file!')

    spectralContrastPCA(scPool, pool)

    # build pitch value histogram
    from math import log
    from numpy import bincount
    # convert from Hz to midi notes
    midipitches = []
    unknown = 0
    for freq in pitches:
        if freq > 0. and freq <= 12600:
            midipitches.append(12*(log(freq/6.875)/0.69314718055995)-3.)
        else:
            unknown += 1

    if len(midipitches) > 0:
      # compute histogram
      midipitchhist = bincount(midipitches)
      # set 0 midi pitch to be the number of pruned value
      midipitchhist[0] = unknown
      # normalise
      midipitchhist = [val/float(sum(midipitchhist)) for val in midipitchhist]
      # zero pad
      for i in range(128 - len(midipitchhist)): midipitchhist.append(0.0)
    else:
      midipitchhist = [0.]*128
      midipitchhist[0] = 1.

    # pitchhist = essentia.array(zip(range(len(midipitchhist)), midipitchhist))
    pool.add(namespace + '.' + 'spectral_pitch_histogram', midipitchhist)#, pool.GlobalScope)

    # the code below is the same as the one above:
    #for note in midipitchhist:
    #    pool.add(namespace + '.' + 'spectral_pitch_histogram_values', note)
    #    print "midi note:", note

    pitch_centralmoments = essentia.CentralMoments(range = len(midipitchhist) - 1)
    (pitch_histogram_spread, pitch_histogram_skewness, pitch_histogram_kurtosis) = distributionshape(pitch_centralmoments(midipitchhist))
    pool.add(namespace + '.' + 'spectral_pitch_histogram_spread', pitch_histogram_spread)#, pool.GlobalScope)

    progress.finish()
