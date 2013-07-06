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

analysisSampleRate = 44100.0

def compute(filename, pool, startTime=0, endTime=1e6,\
            sampleRate=analysisSampleRate):
    '''1st pass: get metadata and replay gain'''

    downmix = 'mix'
    tryReallyHard = True
    while tryReallyHard:
        loader = EqloudLoader(filename=filename,
                              sampleRate=sampleRate,
                              startTime=startTime,
                              endTime=endTime,
                              downmix=downmix)
        rgain = ReplayGain(applyEqloud=False)

        pool.set('metadata.audio_properties.analysis_sample_rate', loader.paramValue('sampleRate'))
        pool.set('metadata.audio_properties.downmix', loader.paramValue('downmix'))

        loader.audio >> rgain.signal
        rgain.replayGain >> (pool, 'metadata.audio_properties.replay_gain')
        try:
            run(loader)
            length = loader.audio.totalProduced()
            tryReallyHard = False
        except:
            if downmix =='mix':
                downmix = 'left'
                try:
                    pool.remove('metadata.audio_properties.downmix')
                    pool.remove('metadata.audio_properties.replay_gain')
                except: pass
                continue
            else:
                INFO('ERROR: ' + filename + ' seems to be a completely silent file... Aborting...')
                sys.exit(1)

        replayGain = pool['metadata.audio_properties.replay_gain']
        if replayGain > 20 :
            if downmix == 'mix':
                downmix = 'left'
                try:
                    pool.remove('metadata.audio_properties.downmix')
                    pool.remove('metadata.audio_properties.replay_gain')
                except: pass
            else:
                INFO('ERROR: ' + filename + 'seems to be a completely silent file... Aborting...')
                sys.exit(1)

    # set duration of audio file:
    pool.set('metadata.audio_properties.length', float(length)/float(sampleRate))



usage = 'replaygain.py <inputfilename> <outputfilename>'

def parse_args():

    import numpy

    essentia_version = '%s\n'\
    'python version: %s\n'\
    'numpy version: %s' % (essentia.__version__,       # full version
                           sys.version.split()[0],     # python major version
                           numpy.__version__)          # numpy version

    from optparse import OptionParser
    parser = OptionParser(usage=usage, version=essentia_version)

    (options, args) = parser.parse_args()

    return options, args



if __name__ == '__main__':

    opts, args = parse_args()

    if len(args) != 2:
        sys.exit(1)
        cmd = './'+os.path.basename(sys.argv[0])+ ' -h'
        os.system(cmd)
        sys.exit(1)

    pool = essentia.Pool()
    compute(args[0], pool, sampleRate=analysisSampleRate)

    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats)(pool)
    essentia.standard.YamlOutput(filename=args[1])(poolStats)
