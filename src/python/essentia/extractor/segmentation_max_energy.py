#!/usr/bin/env python
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

