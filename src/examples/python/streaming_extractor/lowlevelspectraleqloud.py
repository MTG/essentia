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

lowlevelFrameSize = 2048
lowlevelHopSize = 1024
analysisSampleRate = 44100.0

class LowLevelSpectralEqloudExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=lowlevelFrameSize,
                       hopSize=lowlevelHopSize,
                       sampleRate=analysisSampleRate,
                       halfSampleRate=analysisSampleRate*0.5):
        super(LowLevelSpectralEqloudExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         silentFrames='noise')

        # windowing:
        w = Windowing(type='blackmanharris62')
        fc.frame >> w.frame
        # spectrum:
        spec = Spectrum()
        w.frame >> spec.frame
        # spectral centroid:
        square = UnaryOperator(type='square')
        centroid = Centroid(range=halfSampleRate)
        spec.spectrum >> square.array >> centroid.array
        # spectral central moments:
        cm = CentralMoments(range=halfSampleRate)
        ds = DistributionShape()
        spec.spectrum >> cm.array
        cm.centralMoments >> ds.centralMoments
        # dissonance:
        peaks = SpectralPeaks(orderBy='frequency')
        diss = Dissonance()
        spec.spectrum >> peaks.spectrum
        peaks.frequencies >> diss.frequencies
        peaks.magnitudes >> diss.magnitudes
        # spectral contrast:
        sc = SpectralContrast(frameSize=frameSize,
                              sampleRate=sampleRate,
                              numberBands=6,
                              lowFrequencyBound=20,
                              highFrequencyBound=11000,
                              neighbourRatio=0.4,
                              staticDistribution=0.15)
        spec.spectrum >> sc.spectrum

        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        self.outputs['spectral_centroid'] = centroid.centroid
        self.outputs['spectral_kurtosis'] = ds.kurtosis
        self.outputs['spectral_spread'] = ds.spread
        self.outputs['spectral_skewness'] = ds.skewness
        self.outputs['dissonance'] = diss.dissonance
        self.outputs['sccoeffs'] = sc.spectralContrast
        self.outputs['scvalleys'] = sc.spectralValley


usage = 'lowlevelspectraleqloud.py [options] <inputfilename> <outputfilename>'

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
        essentia.translate(LowLevelSpectralEqloudExtractor, 'streaming_extractorlowlevelspectraleqloud', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(LowLevelSpectralEqloudExtractor, 'streaming_extractorlowlevelspectraleqloud', dot_graph=False)

    loader = EqloudLoader(filename=args[0], replayGain=-6)
    lowlevelExtractor = LowLevelSpectralEqloudExtractor()
    pool = essentia.Pool()

    loader.audio >> lowlevelExtractor.signal

    for desc, output in lowlevelExtractor.outputs.items():
        output >> (pool, namespace + '.' + desc)

    essentia.run(loader)

    # compute aggregation on values:
    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats)(pool)

    essentia.standard.YamlOutput(filename=args[1])(poolStats)
