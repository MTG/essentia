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

namespace = 'sfx'

class SfxExtractor(essentia.streaming.CompositeBase):

    def __init__(self):
        super(SfxExtractor, self).__init__()

        envelope = Envelope()
        decrease = Decrease()
        accu = RealAccumulator()
        cm = CentralMoments()
        ds = DistributionShape()
        centroid = Centroid()
        duration = EffectiveDuration()
        log = LogAttackTime()
        decay = StrongDecay()
        flatness = FlatnessSFX()
        max = MaxToTotal()
        tc = TCToTotal()
        der = DerivativeSFX()

        accu.array >> cm.array
        cm.centralMoments >> ds.centralMoments
        envelope.signal >> accu.data
        accu.array >> decrease.array
        accu.array >> centroid.array
        accu.array >> duration.signal
        accu.array >> log.signal
        accu.array >> flatness.envelope
        accu.array >> der.envelope
        envelope.signal >> decay.signal
        envelope.signal >> max.envelope
        envelope.signal >> tc.envelope

        # define inputs:
        self.inputs['signal'] = envelope.signal

        # define outputs:
        self.outputs['temporal_decrease'] = decrease.decrease
        self.outputs['temporal_kurtosis'] = ds.kurtosis
        self.outputs['temporal_spread'] = ds.spread
        self.outputs['temporal_skewness'] = ds.skewness
        self.outputs['temporal_centroid'] = centroid.centroid
        self.outputs['effective_duration'] = duration.effectiveDuration
        self.outputs['logattacktime'] = log.logAttackTime
        self.outputs['strongdecay'] = decay.strongDecay
        self.outputs['flatness'] = flatness.flatness
        self.outputs['max_to_total'] = max.maxToTotal
        self.outputs['tc_to_total'] = tc.TCToTotal
        self.outputs['der_av_after_max'] = der.derAvAfterMax
        self.outputs['max_der_before_max'] = der.maxDerBeforeMax

usage = 'sfx.py [options] <inputfilename> <outputfilename>'

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
        essentia.translate(SfxExtractor, 'streaming_extractorsfx', dot_graph=True)
    elif opts.generate_cpp:
        essentia.translate(SfxExtractor, 'streaming_extractorsfx', dot_graph=False)

    pool = essentia.Pool()
    loader = MonoLoader(filename=args[0])
    sfx = SfxExtractor()

    loader.audio >> sfx.signal
    for desc, output in sfx.outputs.items():
        output >> (pool, namespace + '.' + desc)
    essentia.run(loader)

    # compute aggregation on values:
    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    exceptions = {'lowlevel.mfcc' : ['mean', 'cov', 'icov']}
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats,
                                                 exceptions=exceptions)(pool)

    essentia.standard.YamlOutput(filename=args[1])(poolStats)
