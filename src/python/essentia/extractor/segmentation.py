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


import sys
import numpy
import essentia
import segmentation_bic
import segmentation_max_energy
from essentia import EssentiaError, INFO
from math import *

namespace = 'segmentation'

def print_onset(onset):
    (minutes, seconds) = (int(onset/60.0), int(onset%60))
    print minutes, 'mn', seconds, 's',


def doSegmentation(inputFilename, audio, pool, options):

    # options
    segtype = options[namespace]['type']
    minimumLength = options[namespace]['minimumSegmentsLength']
    writeFile = options[namespace]['writeSegmentsAudioFile']
    sampleRate = options['sampleRate']

    if segtype == 'fromFile':
        segments = [ map(float, l.strip().split('\t')) for l in open(options[namespace]['segmentsFile'], 'r').readlines() ]

    else:
        if segtype == 'maxEnergy':
            onsets = segmentation_max_energy.compute(pool, options)

        elif segtype == 'bic': # Bayesian Information Criterion (Bic) segmentation
            onsets = segmentation_bic.compute(pool, options)

        else:
            raise EssentiaError('Unknown segmentation type: ' + segtype)

        # creating segment wave file
        if writeFile:
            outputFilename = inputFilename + '.segments.wav'
            print 'Creating segments audio file ' + outputFilename + '...'
            audioOnsetsMarker = essentia.AudioOnsetsMarker(filename = outputFilename, sampleRate = sampleRate)
            audioOnsetsMarker(audio, onsets)

        # transforming the onsets into segments = pairs of onsets
        segments = []
        for onset, onsetNext in zip(onsets[:-1], onsets[1:]):
            segments.append([onset, onsetNext])

    if options['verbose']:
        if len(segments) > 0:
            print 'Segments : ',
            for segment in segments:
                print '[',
                print_onset(segment[0])
                print ",",
                print_onset(segment[1])
                print '] ',
        else:
            print 'No segments found!'
        print

    return segments


def compute(inputFilename, audio, pool, options):

    INFO('Doing segmentation...')

    type = options[namespace]['type']
    minimumLength = options[namespace]['minimumSegmentsLength']
    thumbnail = options[namespace]['thumbnailing']

    if pool.value('metadata.duration_processed') < minimumLength:
        segments = []
        INFO('No segments found!')
    else:
        segments = doSegmentation(inputFilename, audio, pool, options)

    #pool.setCurrentNamespace(namespace)
    pool.add(namespace + '.' + 'timestamps', essentia.array(segments))#, pool.GlobalScope)

    return segments

