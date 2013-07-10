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
