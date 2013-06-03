#!/usr/bin/env python

import essentia
from essentia import *
from essentia.streaming import *

analysisSampleRate = 44100.0

def getAnalysisMetadata(pool):
    descriptorNames = pool.descriptorNames()
    if 'metadata.audio_properties.replay_gain' in descriptorNames:
        rgain = pool['metadata.audio_properties.replay_gain']
    else:
        INFO('Warning: replay gain was not found in pool. Using 0.')
        rgain = 0
    if 'metadata.audio_properties.analysis_sample_rate' in descriptorNames:
        sampleRate = pool['metadata.audio_properties.analysis_sample_rate']
    else:
        INFO('Warning: analysis sample rate was not found in pool. Using ', analysisSampleRate)
        sampleRate = analysisSampleRate
    if 'metadata.audio_properties.downmix' in descriptorNames:
        downmix = pool['metadata.audio_properties.downmix']
    else:
        INFO('Warning: downmix was not found in pool. Using \'mix\'')
        downmix = 'mix'
    return rgain, sampleRate, downmix

def readMetadata(filename, pool, failOnError=True):
    metadata = streaming.MetadataReader(filename=filename, failOnError=failOnError)
    metadata.title >> (pool, 'metadata.tags.title')
    metadata.artist >> (pool, 'metadata.tags.artist')
    metadata.album >> (pool, 'metadata.tags.album')
    metadata.comment >> (pool, 'metadata.tags.comment')
    metadata.genre >> (pool, 'metadata.tags.genre')
    metadata.track >> (pool, 'metadata.tags.track')
    metadata.year >> (pool, 'metadata.tags.year')
    metadata.length >> None  # let audio loader take care of this
    metadata.bitrate >> (pool, 'metadata.audio_properties.bitrate')
    metadata.sampleRate >> None # let the audio loader take care of this
    metadata.channels >> (pool, 'metadata.audio_properties.channels')
    run(metadata)


