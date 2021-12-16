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
from essentia import INFO
import sys
import math
from math import *


namespace = 'panning'
dependencies = None


def compute(audio, pool, options):

    INFO('Computing Panning descriptors...')

    filename = pool.descriptors['metadata']['filename']['values'][0]
    sampleRate = options['sampleRate']
    frameSize = options['frameSize']
    hopSize = options['hopSize']

    audioLeft, audioRight, originalSampleRate, originalChannelsNumber = essentia.AudioFileInput(filename = filename,
                                                                                                outputSampleRate = sampleRate,
                                                                                                stereo = 'True')()
    # in case of a mono file
    if originalChannelsNumber == 1:
        audioRight = audioLeft

    panning = essentia.ExtractorPanning(frameSize = frameSize, hopSize = hopSize)
    coefficients = panning(audioLeft, audioRight);

    # used for a nice progress display
    total_frames = len(coefficients)
    n_frames = 0
    start_of_frame = -frameSize*0.5

    progress = essentia.Progress(total = total_frames)

    while n_frames < total_frames:

        frameScope = [ start_of_frame / sampleRate, (start_of_frame + frameSize) / sampleRate ]
        pool.setCurrentScope(frameScope)

        pool.add('coefficients', essentia.array(coefficients[n_frames]), frameScope)

        # display of progress report
        progress.update(n_frames)

        n_frames += 1
        start_of_frame += hopSize

    progress.finish()

