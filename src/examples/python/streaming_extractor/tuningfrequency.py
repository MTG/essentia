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

tonalFrameSize = 4096
tonalHopSize = 2048

class TuningFrequencyExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=tonalFrameSize, hopSize=tonalHopSize):
        super(TuningFrequencyExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         silentFrames='noise')

        w = Windowing(type='blackmanharris62')
        spec = Spectrum()
        peaks = SpectralPeaks(maxPeaks=10000,
                              magnitudeThreshold=0.00001,
                              minFrequency=40,
                              maxFrequency=5000,
                              orderBy='frequency');
        tuning = TuningFrequency()

        fc.frame >> w.frame >> spec.frame
        spec.spectrum >> peaks.spectrum
        peaks.magnitudes >> tuning.magnitudes
        peaks.frequencies >> tuning.frequencies
        tuning.tuningCents >> None

        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        self.outputs['tuningFrequency'] = tuning.tuningFrequency

usage = 'tuningfrequency.py [options] <inputfilename> <outputfilename>'

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
        essentia.translate(TuningFrequencyExtractor, 'streaming_extractortuningfrequency', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(TuningFrequencyExtractor, 'streaming_extractortuningfrequency', dot_graph=False)

    pool = essentia.Pool()
    loader = essentia.streaming.MonoLoader(filename=args[0])
    tuning = TuningFrequencyExtractor()
    loader.audio >> tuning.signal
    tuning.tuningFrequency >> (pool, 'tuning_frequency')
    essentia.run(loader)

    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats)(pool)
    essentia.standard.YamlOutput(filename=args[1])(poolStats)
