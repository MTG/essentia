#!/usr/bin/env python

from essentia import *

import replaygain
import lowlevel
import sys

analysisSampleRate = 44100.0
lowlevelHopSize = 1024 # hopsize used to compute lowlevel features

def compute(filename, pool, startTime=0, endTime=1e6):

    if not pool.containsKey('lowlevel.mfcc'):
        INFO('Error: mfcc must be computed prior to segmentation')
        sys.exit(1)

    # parameters for sbic:
    minimumSegmentsLength = 10 # seconds
    size1 = 1000
    inc1 = 300
    size2 = 600
    inc2 = 50
    cpw = 5

    # TODO: don't know why we need to copy mfcc to a list, but otherwise the
    # SBic algorithm returns the wrong number of segments
    # features = pool['lowlevel.mfcc'].transpose() # will not work correctly
    features = [val for val in pool['lowlevel.mfcc'].transpose()]


    sbic = standard.SBic(size1=size1, inc1=inc1,
                         size2=size2, inc2=inc2,
                         cpw=cpw, minLength=minimumSegmentsLength)

    # only BIC segmentation at the moment:
    segments = sbic(array(features))
    if pool.containsKey('metadata.audio_properties.analysis_sample_rate'):
        sampleRate = pool['metadata.audio_properties.analysis_sample_rate']
    else:
        sampleRate = analysisSampleRate
        INFO('Warning: sample rate not found in pool. Using default: '+analysisSampleRate)
    for segment in segments:
        pool.add('segmentation.timestamps', segment*lowlevelHopSize/sampleRate)
