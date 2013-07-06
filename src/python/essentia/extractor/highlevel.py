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


namespace = 'highlevel'
dependencies = [ 'lowlevel', 'tempotap', 'beats' ]


def compute(audio, pool, options):
    defaultStats=['mean', 'min', 'max', 'var', 'dmean', 'dvar', 'dmean2', 'dvar2', 'value']
    aggPool = essentia.PoolAggregator(defaultStats=defaultStats)(pool)
    descriptors = aggPool.descriptorNames()
    profile = 'music'

    INFO('Computing High-Level descriptors...')

    if profile == 'music':
        # Excitement
        excitement(aggPool)
        # Excitement
        intensity(aggPool)

    INFO('100% done...')

def excitement(pool):

    # this describes if a song is exciting or not on 3 levels: 1 (not exciting), 2 or 3 (very exciting)
    spectral_centroid_mean = pool.value('lowlevel.spectral_centroid.mean')
    tempotap_bpm_value = pool.value('rhythm.bpm.value')
    rhythm_beats_loudness_mean = pool.value('rhythm.beats_loudness.mean')
    rhythm_onset_rate_value = pool.value('rhythm.onset_rate.value')

    # Weka tree J48 calculated with essentia_0.4.0:2875
    if spectral_centroid_mean <= 2254.374756:
       if rhythm_onset_rate_value <= 4.521962:
          if spectral_centroid_mean <= 1932.181519:
             excitement = 1
          else:
             if rhythm_beats_loudness_mean <= 0.032491:
                excitement = 1
             else:
                excitement = 2
       else:
          if rhythm_beats_loudness_mean <= 0.051655:
             excitement = 3
          else:
             excitement = 2
    else:
       if spectral_centroid_mean <= 2477.170654:
          excitement = 2
       else:
          if tempotap_bpm_value <= 128.839981:
             if rhythm_beats_loudness_mean <= 0.041298:
                excitement = 3
             else:
                excitement = 2
          else:
             excitement = 3

    pool.add(namespace + '.' + 'excitement', excitement)#, pool.GlobalScope)

def intensity(pool):

    # this describes if a song is intense or not: from 0 to 1
    tempotap_bpm_value = pool.value('rhythm.bpm.value')
    rhythm_onset_rate_value = pool.value('rhythm.onset_rate.value')
    rhythm_beats_loudness_mean = pool.value('rhythm.beats_loudness.mean')
    rhythm_beats_loudness_bass_mean = pool.value('rhythm.beats_loudness_bass.mean')

    intensity = 0

    # this algorithm is based on the common sense
    # the thresholds were found from essentia_0.4.0:2885
    if tempotap_bpm_value < 100.0:
       intensity += 1
    else:
       if tempotap_bpm_value < 120.0:
          intensity += 2
       else:
          intensity += 3

    if rhythm_onset_rate_value < 3.0:
       intensity += 1
    else:
       if rhythm_onset_rate_value < 5.0:
          intensity += 2
       else:
          intensity += 3

    if rhythm_beats_loudness_mean < 0.1:
       intensity += 1
    else:
       if rhythm_beats_loudness_mean < 0.2:
          intensity += 2
       else:
          intensity += 3

    if rhythm_beats_loudness_bass_mean < 0.2:
       intensity += 1
    else:
       if rhythm_beats_loudness_bass_mean < 0.4:
          intensity += 2
       else:
          intensity += 3

    intensity /= 12.0

    pool.add(namespace + '.' + 'intensity', intensity)#, pool.GlobalScope)

