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


