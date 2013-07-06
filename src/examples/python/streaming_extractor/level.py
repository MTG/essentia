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

#! /usr/bin/env python

import sys, os
import essentia, essentia.standard, essentia.streaming
from essentia.streaming import *
from numpy import argmax, log10, mean, tanh

dynamicFrameSize = 88200
dynamicHopSize = 44100
analysisSampleRate = 44100.0

# expects the audio source to already be equal-loudness filtered
class LevelExtractor(essentia.streaming.CompositeBase):
    #"""describes the dynamics of an audio signal"""

    def __init__(self, frameSize=dynamicFrameSize, hopSize=dynamicHopSize):
        super(LevelExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         startFromZero=True,
                         silentFrames='noise')

        dy = Loudness()
        fc.frame >> dy.signal

        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        self.outputs['loudness'] = dy.loudness

def squeezeRange(x, x1, x2):
    return 0.5 + 0.5 * tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1))

def levelAverage(pool, namespace=''):
    epsilon = 1e-4
    threshold = 1e-4 # -80dB

    if namespace: namespace += '.lowlevel.'
    else: namespace = 'lowlevel.'
    loudness = pool[namespace + 'loudness']
    pool.remove(namespace + 'loudness')
    maxValue = loudness[argmax(loudness)]
    if maxValue <= epsilon: maxValue = epsilon

    # normalization of the maximum:
    def f(x):
        x /= float(maxValue)
        if x <= threshold : return threshold
        return x

    loudness = map(f, loudness)

    # average level:
    levelAverage = 10.0*log10(mean(loudness))

    # Re-scaling and range-control
    # This yields in numbers between
    # 0 for signals with  large dynamic variace and thus low dynamic average
    # 1 for signal with little dynamic range and thus
    # a dynamic average close to the maximum
    x1 = -5.0
    x2 = -2.0
    levelAverageSqueezed = squeezeRange(levelAverage, x1, x2)
    pool.set(namespace + 'average_loudness', levelAverageSqueezed)



usage = 'level.py [options] <inputfilename> <outputfilename>'

def parse_args():

    import numpy
    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=usage, version=essentia_version)

    parser.add_option("-c","--cpp", action="store_true", dest="generate_cpp",
      help="generate cpp code from CompositeBase algorithm")

    parser.add_option("-d", "--dot", action="store_true", dest="generate_dot",
      help="generate dot and cpp code from CompositeBase algorithm")

    (options, args) = parser.parse_args()

    return options, args



if __name__ == '__main__':

    opts, args = parse_args()

    if len(args) != 2:
        cmd = './'+os.path.basename(sys.argv[0])+ ' -h'
        os.system(cmd)
        sys.exit(1)

    if opts.generate_dot:
        essentia.translate(LevelExtractor, 'streaming_extractorlevel', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(LevelExtractor, 'streaming_extractorlevel', dot_graph=False)

    # find out replay gain:
    loader = EqloudLoader(filename=args[0],
                          sampleRate=analysisSampleRate,
                          downmix='mix')
    rgain = ReplayGain(applyEqloud=False)

    pool = essentia.Pool()

    loader.audio >> rgain.signal
    rgain.replayGain >> (pool, 'replay_gain')
    essentia.run(loader)

    # get average level:
    loader = EqloudLoader(filename=args[0],
                          replayGain=pool['replay_gain'],
                          sampleRate=analysisSampleRate,
                          downmix='mix')

    levelExtractor = LevelExtractor()
    loader.audio >> levelExtractor.signal
    levelExtractor.loudness >> (pool, 'lowlevel.loudness')
    essentia.run(loader)

    levelAverage(pool)

    essentia.standard.YamlOutput(filename=args[1])(pool)
