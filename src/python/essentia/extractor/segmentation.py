#!/usr/bin/env python
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

