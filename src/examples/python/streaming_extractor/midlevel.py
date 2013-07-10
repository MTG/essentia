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



import sys, os

import essentia
from essentia import *
from essentia.streaming import *

from metadata import getAnalysisMetadata
from tonaldescriptors import TonalDescriptorsExtractor

analysisSampleRate = 44100.0

def compute(source, pool, startTime=0, endTime=1e6, namespace='',
            sampleRate=analysisSampleRate):
    '''3rd pass: HPCP & beats loudness (depend on some descriptors that
              have been computed during the 2nd pass)'''

    if namespace:
        tonalspace = namespace + '.tonal.'
        rhythmspace = namespace + '.rhythm.'
    else:
        tonalspace = 'tonal.'
        rhythmspace = 'rhythm.'

    tuning_frequency = 440.0
    if pool.containsKey(tonalspace + 'tuning_frequency'):
        tuning_frequency = pool[tonalspace + 'tuning_frequency'][-1]
    tonalDescriptors = TonalDescriptorsExtractor(tuningFrequency=tuning_frequency)
    source >> tonalDescriptors.signal
    for desc, output in tonalDescriptors.outputs.items():
        output >> (pool, tonalspace + desc)

    ticks = pool[rhythmspace + 'beats_position']
    beatsLoudness = streaming.BeatsLoudness(sampleRate=sampleRate,
                                            beats=ticks)
    source >> beatsLoudness.signal
    beatsLoudness.loudness >> (pool, rhythmspace + 'beats_loudness')
    beatsLoudness.loudnessBass >> (pool, rhythmspace + 'beats_loudness_bass')


usage = 'midlevel.py [options] <inputfilename> <outputfilename>'

def parse_args():

    import numpy

    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=usage, version=essentia_version)

    parser.add_option("--start",
                      action="store", dest="startTime", default="0.0",
                      help="time in seconds from which the audio is computed")

    parser.add_option("--end",
                      action="store", dest="endTime", default="600.0",
                      help="time in seconds till which the audio is computed, 'end' means no time limit")


    (options, args) = parser.parse_args()

    return options, args



if __name__ == '__main__':

    from metadata import readMetadata, getAnalysisMetadata
    import replaygain
    import lowlevel

    opts, args = parse_args()

    if len(args) != 2:
        sys.exit(1)
        cmd = './'+os.path.basename(sys.argv[0])+ ' -h'
        os.system(cmd)
        sys.exit(1)

    startTime = float(opts.startTime)
    endTime=float(opts.endTime)
    pool = essentia.Pool()
    readMetadata(args[0], pool)
    rgain, sampleRate, downmix = getAnalysisMetadata(pool)
    loader = streaming.EqloudLoader(filename = filename,
                                    sampleRate = sampleRate,
                                    startTime = startTime,
                                    endTime = endTime,
                                    replayGain = rgain,
                                    downmix = downmix)

    replaygain.compute(source, pool, startTime, endTime)
    lowlevel.compute(source, pool, startTime, endTime,
                    sampleRate=analysisSampleRate)
    compute(source, pool, startTime, endTime,
                    sampleRate=analysisSampleRate)

    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats)(pool)
    essentia.standard.YamlOutput(filename=args[1])(poolStats)
