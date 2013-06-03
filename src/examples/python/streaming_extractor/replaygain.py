#!/usr/bin/env python

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
