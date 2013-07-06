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

analysisSampleRate = 44100.0

lowlevelFrameSize = 2048
lowlevelHopSize = 1024
range = 1

class PitchExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=lowlevelFrameSize, hopSize=lowlevelHopSize):
        super(PitchExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         silentFrames='noise')
        w = Windowing(type='blackmanharris62')
        fc.frame >> w.frame
        spec = Spectrum()
        w.frame >> spec.frame
        pitch = PitchDetection(frameSize=frameSize)
        spec.spectrum >> pitch.spectrum
        pitch.pitchConfidence >> None

        self.inputs['signal'] = fc.signal
        self.outputs['pitch'] = pitch.pitch



usage = 'pitch.py [options] <inputfilename> <outputfilename>'

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
        essentia.translate(PitchExtractor, 'streaming_extractorpitch', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(PitchExtractor, 'streaming_extractorpitch', dot_graph=False)

    # find out replay gain
    loader = EqloudLoader(filename=args[0],
                          sampleRate=analysisSampleRate,
                          downmix='mix')
    rgain = ReplayGain(applyEqloud=False)

    pool = essentia.Pool()

    loader.audio >> rgain.signal
    rgain.replayGain >> (pool, 'replay_gain')
    essentia.run(loader)

    # pitch detection
    loader2 = EqloudLoader(filename=args[0],
                           replayGain=pool['replay_gain'],
                           sampleRate=analysisSampleRate,
                           downmix='mix')

    pitch = PitchExtractor()
    loader2.audio >> pitch.signal
    pitch.pitch >> (pool, 'lowlevel.pitch')
    essentia.run(loader2)


    essentia.standard.YamlOutput(filename=args[1])(pool)
