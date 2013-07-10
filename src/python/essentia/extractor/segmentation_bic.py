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
import os
from essentia import EssentiaError
import math
from math import *
import pdb

namespace = 'lowlevel'
dependencies = ['lowlevel']

def compute(pool, options):

    namespaces = [desc.split('.')[0] for desc in pool.descriptorNames()]


    if namespace not in namespaces:
        print 'ERROR when trying to compute BIC segmentation: you must compute lowlevel descriptors first!'

    # options
    minimumSegmentsLength = options['segmentation']['minimumSegmentsLength']
    sampleRate = options['sampleRate']
    lowlevel = options['segmentation']['bicSettings']['lowlevel']
    cpw = options['segmentation']['bicSettings']['cpw']
    size1 = options['segmentation']['bicSettings']['size1']
    inc1 = options['segmentation']['bicSettings']['inc1']
    size2 = options['segmentation']['bicSettings']['size2']
    inc2 = options['segmentation']['bicSettings']['inc2']

    # calculate minimumLength
    #for k,v in options.items():
    #    print k, ':', v
    #print options['specific'][namespace]['hopSize'], options['sampleRate']
    #for name in lowlevel:
    #    hopSize = pool.descriptors[namespace][name]['scopes'][1][0] - pool.descriptors[namespace][name]['scopes'][0][0]
    hopSize = pool.value(namespace + '.' + 'scope')[1][0] - pool.value(namespace + '.' + 'scope')[0][0]
    minimumLength = int(minimumSegmentsLength / hopSize+0.5)

    # BIC segmentation algorithm
    bic = essentia.SBic(cpw = cpw, size1 = size1, inc1 = inc1, size2 = size2, inc2 = inc2, minLength = minimumLength)

    # create descriptor values array
    descriptors = []

    for name in lowlevel:

        descScopes = pool.value(namespace+ '.' + 'scope')
        descValues = pool.value(namespace + '.' +  name)
        try:
            # special case: descriptors with more than one dimension (mfcc, barkbands, etc...)
            for i in range(len(descValues[0])):
                descSubValues = []
                for value in descValues:
                    descSubValues.append(value[i])
                descriptors.append(descSubValues)
        except:
            descriptors.append(descValues)

    descriptorArray = essentia.array(descriptors)

    # compute the segmentation
    segments = bic(descriptorArray)

    # compute onsets
    segments.sort()
    onsets = []
    for s in segments:
        onsets.append(descScopes[int(s)][1])
    onsets.sort()

    # we don't need these scopes anymore
    pool.remove(namespace + '.' + 'scopes')

    return onsets

