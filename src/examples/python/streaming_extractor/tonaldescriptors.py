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
import tuningfrequency

tonalFrameSize = 4096
tonalHopSize = 2048

class TonalDescriptorsExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=tonalFrameSize, hopSize=tonalHopSize, tuningFrequency=440.0):
        super(TonalDescriptorsExtractor, self).__init__()

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize,
                         silentFrames='noise')

        w = Windowing(type='blackmanharris62')
        spec = Spectrum()
        peaks = SpectralPeaks(maxPeaks=10000,
                              magnitudeThreshold=0.00001,
                              minFrequency=40,
                              maxFrequency=5000,
                              orderBy='magnitude');
        hpcp_key = HPCP(size = 36,
                        referenceFrequency = tuningFrequency,
                        bandPreset = False,
                        minFrequency = 40.0,
                        maxFrequency = 5000.0,
                        weightType = 'squaredCosine',
                        nonLinear = False,
                        windowSize = 4.0/3.0);
        key = Key()
        hpcp_chord = HPCP(size = 36,
                          referenceFrequency = tuningFrequency,
                          harmonics=8,
                          bandPreset = True,
                          minFrequency = 40.0,
                          maxFrequency = 5000.0,
                          splitFrequency = 500.0,
                          weightType = 'cosine',
                          nonLinear = True,
                          windowSize = 0.5);
        chords = ChordsDetection()
        chords_desc = ChordsDescriptors()
        hpcp_tuning = HPCP(size = 120,
                           referenceFrequency = tuningFrequency,
                           harmonics=8,
                           bandPreset = True,
                           minFrequency = 40.0,
                           maxFrequency = 5000.0,
                           splitFrequency = 500.0,
                           weightType = 'cosine',
                           nonLinear = True,
                           windowSize = 0.5);

        fc.frame >> w.frame >> spec.frame
        spec.spectrum >> peaks.spectrum
        peaks.frequencies >> hpcp_key.frequencies
        peaks.magnitudes >> hpcp_key.magnitudes
        hpcp_key.hpcp >> key.pcp
        peaks.frequencies >> hpcp_chord.frequencies
        peaks.magnitudes >> hpcp_chord.magnitudes
        hpcp_chord.hpcp >> chords.pcp
        chords.chords >> chords_desc.chords
        key.key >> chords_desc.key
        key.scale >> chords_desc.scale
        peaks.frequencies >> hpcp_tuning.frequencies
        peaks.magnitudes >> hpcp_tuning.magnitudes



        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        self.outputs['hpcp'] = hpcp_key.hpcp
        self.outputs['key_key'] = key.key
        self.outputs['key_scale'] = key.scale
        self.outputs['key_strength'] = key.strength
        self.outputs['chords_progression'] = chords.chords
        self.outputs['chords_strength'] = chords.strength
        self.outputs['chords_histogram'] = chords_desc.chordsHistogram
        self.outputs['chords_number_rate'] = chords_desc.chordsNumberRate
        self.outputs['chords_changes_rate'] = chords_desc.chordsChangesRate
        self.outputs['chords_key'] = chords_desc.chordsKey
        self.outputs['chords_scale'] = chords_desc.chordsScale
        self.outputs['hpcp_highres'] = hpcp_tuning.hpcp

usage = 'tonaldescriptors.py [options] <inputfilename> <outputfilename>'

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
        essentia.translate(TonalDescriptorsExtractor, 'streaming_extractortonaldescriptors', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(TonalDescriptorsExtractor, 'streaming_extractortonaldescriptors', dot_graph=False)

    pool = essentia.Pool()
    loader = essentia.streaming.MonoLoader(filename=args[0])
    tonalExtractor = TonalDescriptorsExtractor()
    loader.audio >> tonalExtractor.signal
    for desc, output in tonalExtractor.outputs.items():
        output >> (pool, desc)
    essentia.run(loader)

    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats)(pool)
    essentia.standard.YamlOutput(filename=args[1])(poolStats)
