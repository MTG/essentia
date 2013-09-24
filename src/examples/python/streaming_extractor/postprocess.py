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
import essentia.standard as standard
import essentia.streaming as streaming
from essentia import Pool
from numpy import mean, var

def pca(pool, namespace=''):
    llspace = 'lowlevel.'
    if namespace: llspace = namespace + '.lowlevel.'
    sccoeffs = pool[llspace + 'sccoeffs']
    scvalleys = pool[llspace + 'scvalleys']
    numFrames = len(sccoeffs)
    poolSc = Pool()
    merged = essentia.zeros(2*len(sccoeffs[0]))
    for frame in xrange(numFrames):
        j = 0
        for i in xrange(len(sccoeffs[frame])):
            merged[j]=sccoeffs[frame][i]
            merged[j+1]=scvalleys[frame][i]
            j+=2
        poolSc.add('contrast', merged)

    poolTransformed = standard.PCA(namespaceIn='contrast',
                                   namespaceOut='contrast')(poolSc)

    contrast = poolTransformed['contrast']

    pool.set(llspace+'spectral_contrast.mean', mean(contrast, axis=0))
    pool.set(llspace+'spectral_contrast.var', var(contrast, axis=0))

    pool.remove(llspace+'sccoeffs')
    pool.remove(llspace+'scvalleys')

def postProcess(pool, namespace=''):
 # Add missing descriptors which are not computed yet, but will be for the
 # final release or during the 1.x cycle. However, the schema need to be
 # complete before that, so just put default values for these.
 # Also make sure that some descriptors that might have fucked up come out nice.

    rhythmspace='rhythm.'
    if namespace:
        rhythmspace=namespace+'.rhythm.'

    pool.set(rhythmspace + 'bpm_confidence', 0.0);
    pool.set(rhythmspace + 'perceptual_tempo', 'unknown');

    descriptors = pool.descriptorNames()
    if not pool.containsKey(rhythmspace+'beats_loudness'):
      pool.set(rhythmspace + 'beats_loudness', 0.0);
      pool.set(rhythmspace + 'beats_loudness_bass', 0.0);

    # PCA analysis of spectral contrast output:
    pca(pool, namespace);
