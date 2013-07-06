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
from lowlevelspectral import LowLevelSpectralExtractor
from lowlevelspectraleqloud import LowLevelSpectralEqloudExtractor
from level import LevelExtractor
from tuningfrequency import TuningFrequencyExtractor
from rhythmdescriptors import RhythmDescriptorsExtractor


analysisSampleRate = 44100.0

def compute(eqloudSource, neqloudSource, pool, startTime=0, endTime=1e6, namespace='',
            sampleRate=analysisSampleRate):
    '''2nd pass: normalize the audio with replay gain, compute as
             many lowlevel descriptors as possible'''

    llspace = 'lowlevel.'
    rhythmspace = 'rhythm.'
    sfxspace = 'sfx.'
    tonalspace = 'tonal.'

    if namespace:
        llspace = namespace + '.lowlevel.'
        rhythmspace = namespace + '.rhythm.'
        sfxspace = namespace + '.sfx.'
        tonalspace = namespace + '.tonal.'

    # Low-Level Spectral Descriptors
    lowLevelSpectral = LowLevelSpectralExtractor(halfSampleRate=sampleRate*0.5)
    neqloudSource >> lowLevelSpectral.signal
    sfx_descriptors = ["inharmonicity", "oddtoevenharmonicenergyratio", "tristimulus"]
    for desc, output in lowLevelSpectral.outputs.items():
        if desc in sfx_descriptors:
            output >> (pool, sfxspace + desc)
        else: output >> (pool, llspace + desc)

    # Low-Level Spectral Equal Loudness Descriptors
    lowLevelSpectralEqloud = LowLevelSpectralEqloudExtractor(sampleRate=sampleRate)
    eqloudSource >> lowLevelSpectralEqloud.signal
    for desc, output in lowLevelSpectralEqloud.outputs.items():
        output >> (pool, llspace + desc)

    # Level Descriptor
    level = LevelExtractor()
    eqloudSource >> level.signal
    level.loudness >> (pool, llspace + 'loudness')

    # Tuning Frequency
    tuningFrequency = TuningFrequencyExtractor()
    neqloudSource >> tuningFrequency.signal
    tuningFrequency.tuningFrequency >> (pool, tonalspace + 'tuning_frequency')

    # Rhythm descriptors
    rhythm = RhythmDescriptorsExtractor()
    neqloudSource >> rhythm.signal
    for desc, output in rhythm.outputs.items():
        output >> (pool, rhythmspace + desc)

    # onset detection:
    onsets = streaming.OnsetRate()
    neqloudSource >> onsets.signal
    onsets.onsetTimes >> (pool, rhythmspace + 'onset_times')
    onsets.onsetRate >> None # computed later



usage = 'lowlevel.py [options] <inputfilename> <outputfilename>'

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

    from metadata import readMetadata
    import replaygain

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
    replaygain.compute(args[0], pool, startTime, endTime)
    rgain, sampleRate, downmix = getAnalysisMetadata(pool)

    loader = streaming.EqloudLoader(filename = args[0],
                                    sampleRate = sampleRate,
                                    startTime = startTime,
                                    endTime = endTime,
                                    replayGain = rgain,
                                    downmix = downmix)

    compute(loader.audio, loader.audio, pool, startTime, endTime,
            sampleRate=analysisSampleRate)

    essentia.run(loader)

    # check if we processed enough audio for it to be useful, in particular did
    # we manage to get an estimation for the loudness (2 seconds required)
    if not pool.containsKey(llspace + "loudness"):
        INFO('ERROR: File is too short (< 2sec)... Aborting...')
        sys.exit(2)

    numOnsets = len(pool[rhythmspace + 'onset_times'])
    sampleRate = pool['metadata.audio_properties.analysis_sample_rate']
    onset_rate = numOnsets/float(source.totalProduced())*sampleRate
    pool.set(rhythmspace + 'onset_rate', onset_rate)




    stats = ['mean', 'var', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']
    poolStats = essentia.standard.PoolAggregator(defaultStats=stats)(pool)
    essentia.standard.YamlOutput(filename=args[1])(poolStats)
