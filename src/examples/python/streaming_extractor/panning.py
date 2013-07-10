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

namespace = 'lowlevel'
panningFrameSize = 8192
panningHopSize = 2048
analysisSampleRate = 44100.0


class PanningExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=panningFrameSize, hopSize=panningHopSize,
                       sampleRate=analysisSampleRate):
        super(PanningExtractor, self).__init__()

        demuxer = StereoDemuxer()
        fc_left = FrameCutter(frameSize=frameSize,
                              hopSize=hopSize,
                              startFromZero=False,
                              silentFrames='noise')

        fc_right = FrameCutter(frameSize=frameSize,
                               hopSize=hopSize,
                               startFromZero=False,
                               silentFrames='noise')
        w_left = Windowing(type='hann',
                           size=frameSize,
                           zeroPadding=frameSize)
        w_right = Windowing(type='hann',
                            size=frameSize,
                            zeroPadding=frameSize)
        spec_left = Spectrum()
        spec_right = Spectrum()
        pan = Panning(sampleRate=sampleRate,
                      averageFrames=43, # 2 seconds*sr/hopsize
                      panningBins=512,
                      numCoeffs=20,
                      numBands=1,
                      warpedPanorama=True)

        # left channel:
        demuxer.left >> fc_left.signal
        fc_left.frame >> w_left.frame >> spec_left.frame
        spec_left.spectrum >> pan.spectrumLeft

        # right channel:
        demuxer.right >> fc_right.signal
        fc_right.frame >> w_right.frame >> spec_right.frame
        spec_right.spectrum >> pan.spectrumRight

        # define inputs:
        self.inputs['signal'] = demuxer.audio

        # define outputs:
        self.outputs['panning_coefficients'] = pan.panningCoeffs


def computePanning(filename, pool, startTime, endTime, namespace=''):
    '''4th pass: panning'''

    if namespace:
        llspace = namespace + '.lowlevel.'
    else:
        llspace = 'lowlevel.'

    rgain, analysisSampleRate, downmix = getAnalysisMetadata(pool)
    loader = streaming.AudioLoader(filename=filename)
    trimmer = streaming.Trimmer(startTime=startTime, endTime=endTime)
    panning = PanningExtractor(sampleRate=analysisSampleRate)
    loader.audio >> panning.signal
    loader.sampleRate >> None
    loader.numberChannels >> None
    panning.panning_coefficients >> (pool,'panning_coefficients')
    run(loader)



usage = 'panning.py [options] <inputfilename> <outputfilename>'


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
        essentia.translate(PanningExtractor, 'streaming_extractorpanning', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(PanningExtractor, 'streaming_extractorpanning', dot_graph=False)

    pool = essentia.Pool()
    loader = AudioLoader(filename=args[0])
    panExtractor = PanningExtractor()

    loader.audio >> panExtractor.signal
    loader.numberChannels >> None
    loader.sampleRate >> None
    panExtractor.panning_coefficients >> (pool, namespace + '.panning_coefficients')
    essentia.run(loader)

    essentia.standard.YamlOutput(filename=args[1])(pool)
