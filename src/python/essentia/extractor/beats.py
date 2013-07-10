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
from essentia import INFO
import sys
import math
from math import *
from essentia.progress import Progress


namespace = 'rhythm'
dependencies = None

def max_energy_index(beat):

    energy = []

    for sample in beat:
        energy.append(sample * sample)

    return energy.index(max(energy))


def compute(audio, pool, options):

    INFO("Computing Beats descriptors...")

    sampleRate = options['sampleRate']
    windowType = options['windowType']

    beat_window_duration = 0.1 # 100ms
    beat_duration = 0.05 # 50ms estimation after checking some drums kicks duration on freesound


    beats = pool.value('rhythm.beats_position')[0]

    # special case
    if len(beats) == 0:

        # we add them 2 times to get 'mean/var' stats and not 'value'
        # and not on full scope so it's not global
        # FIXME: should use "undefined"
        pool.add("beats_loudness", 0.0,      [0., 0.])
        pool.add("beats_loudness", 0.0,      [0., 0.])
        pool.add("beats_loudness_bass", 0.0, [0., 0.])
        pool.add("beats_loudness_bass", 0.0, [0., 0.])

        INFO('100% done...')

        return

    duration = pool.value('metadata.duration_processed')[0]

    # FIXME: converted to samples in order to have more accurate control of the size of
    # the window. This is due to FFT not being able to be computed on arrays of
    # odd sizes. Please FIXME later, when FFT accepts all kinds of sizes.
    beat_window_duration = int(beat_window_duration*float(sampleRate) + 0.5)
    beat_duration = int(beat_duration*float(sampleRate) + 0.5)
    duration *= float(sampleRate)
    if beat_duration%2 == 1:
        beat_duration += 1;
        beat_window_duration = beat_duration*2;

    energy = essentia.Energy()
    energybandratio = essentia.EnergyBandRatio(startFrequency = 20.0, stopFrequency = 150.0, sampleRate = sampleRate)

    total_beats = len(beats)
    n_beats = 1

    progress = Progress(total = total_beats)

    between_beats_start = [0.0]
    between_beats_end = []

    beats_spectral_energy = 0.0

    # love on the beats
    for beat in beats:
        # convert beat to samples in order to ensure an even size
        beat = beat*float(sampleRate)

        beat_window_start = (beat - beat_window_duration / 2.0) # in samples
        beat_window_end = (beat + beat_window_duration / 2.0) # in samples

        if beat_window_start > 0.0 and beat_window_end < duration: # in samples
            #print "duration: ", duration, "start:", beat_window_start, "end:", beat_window_end

            beat_window = audio[beat_window_start : beat_window_end]

            beat_start = beat_window_start + max_energy_index(beat_window)
            beat_end = beat_start + beat_duration
            beat_audio = audio[beat_start : beat_end]

            beat_scope = [beat_start / float(sampleRate), beat_end/float(sampleRate)] # in seconds
            #print "beat audio size: ", len(beat_audio)

            window = essentia.Windowing(size = len(beat_audio), zeroPadding = 0, type = windowType)
            spectrum = essentia.Spectrum(size = len(beat_audio))
            beat_spectrum = spectrum(window(beat_audio))

            beat_spectral_energy = energy(beat_spectrum)
            pool.add(namespace + '.' + 'beats_loudness', beat_spectral_energy)#, beat_scope)
            beats_spectral_energy += beat_spectral_energy

            beat_spectral_energybandratio = energybandratio(beat_spectrum)
            pool.add(namespace + '.' + 'beats_loudness_bass', beat_spectral_energybandratio)#, beat_scope)

            # filling between-beats arrays
            between_beats_end.append(beat_start/float(sampleRate))
            between_beats_start.append(beat_end/float(sampleRate))

        # display of progress report
        progress.update(n_beats/float(sampleRate))

        n_beats += 1

    between_beats_end.append(duration)

    between_beats_spectral_energy = 0.0

    # love in between beats
    '''
    for between_beat_start, between_beat_end in zip(between_beats_start, between_beats_end):

        between_beat_audio = audio[between_beat_start * sampleRate : between_beat_end * sampleRate]

        between_beat_scope = [between_beat_start, between_beat_end]

        window = essentia.Windowing(windowSize = len(between_beat_audio), zeroPadding = 0, type = "blackmanharris62")
        spectrum = essentia.Spectrum(size = len(between_beat_audio))
        between_beat_spectrum = spectrum(window(between_beat_audio))

        between_beat_spectral_energy = energy(between_beat_spectrum)
        between_beats_spectral_energy += between_beat_spectral_energy
    '''

    progress.finish()
