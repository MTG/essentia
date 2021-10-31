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

import essentia
import numpy
import sys
from math import *
from essentia import INFO
from squeezeInto import squeezeInto
from essentia.essentia_extractor import descriptorNames
from essentia.progress import Progress

namespace = 'lowlevel'
dependencies = None

def compute(audio, pool, options):

    # analysis parameters
    sampleRate = options['sampleRate']
    frameSize  = options['frameSize']
    hopSize    = options['hopSize']
    windowType = options['windowType']

    # frame algorithms
    frames = essentia.FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize, startFromZero = True)
    loudness = essentia.Loudness()

    INFO('Computing Dynamic descriptors...')

    # used for a nice progress display
    total_frames = frames.num_frames()
    n_frames = 0

    level_array = []

    progress = Progress(total = total_frames)

    for frame in frames:

        frame_level = loudness(frame)
        level_array.append(frame_level)

        # display of progress report
        progress.update(n_frames)

        n_frames += 1

    # Maximum dynamic
    EPSILON = 10e-6
    max_value = max(level_array)
    if max_value <= EPSILON:
       max_value = EPSILON

    # Normalization to the maximum
    THRESHOLD = 0.0001 # this corresponds to -80dB
    for i in range(len(level_array)):
      level_array[i] /= max_value
      if level_array[i] <= THRESHOLD:
         level_array[i] = THRESHOLD

    # Dynamic Average
    mean = essentia.Mean()
    average_loudness = 10.0*log10(mean(level_array))

    # re-scaling and range-control
    # This yields in numbers between
    #
    #  0 for signals with  large dynamic variace and
    #    thus low dynamic average
    #  1 for signal with little dynamic range and thus
    # a dynamic average close to the maximum

    # TO DO: [0, 0] should be pool.GlobalScope
    average_loudness_within_zero_to_one = squeezeInto([-5, 0], [-2, 1], average_loudness)
    pool.add(namespace + "." + "average_loudness", average_loudness_within_zero_to_one)#, pool.GlobalScope)

    # Dynamic Fluctuation
    '''
    variance = essentia.Variance()
    level_variance = variance(level_array)
    if level_variance <= THRESHOLD:
       level_variance = THRESHOLD
    level_fluctuation = 10*log10(level_variance)
    # TO DO: [0, 0] should be pool.GlobalScope
    pool.add("level_fluctuation", level_fluctuation, pool.GlobalScope)
    '''

    INFO('\r100% done...')


def postProcess(value):
    return value
