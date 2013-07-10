#!/usr/bin/env python

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


import sys
import numpy
import essentia
from essentia import EssentiaError

dependencies = ['lowlevel']

def compute(pool, options):

    minimumLength = options['segmentation']['minimumSegmentsLength']
    energy = pool.descriptors['lowlevel']['spectral_rms']['values']
    scopes = pool.descriptors['lowlevel']['spectral_rms']['scopes']

    energyScopes = []
    for scope in scopes:
        energyScopes.append(scope[0])

    energyHop = energyScopes[1] - energyScopes[0]

    hopSizeDuration = 1.0
    hopSize = hopSizeDuration / energyHop
    frameSizeDuration = minimumLength
    frameSize = frameSizeDuration / energyHop

    frames = essentia.FrameGenerator(audio = energy, frameSize = frameSize, hopSize = hopSize, startFromZero = True)

    framesEnergy = []

    for frame in frames:
        framesEnergy.append(sum(frame))

    maxFrameIndex = framesEnergy.index(max(framesEnergy))
    onsetStart = energyScopes[int(maxFrameIndex * hopSize)]
    onsetEnd = onsetStart + frameSizeDuration

    onsets = [onsetStart, onsetEnd]

    return onsets

