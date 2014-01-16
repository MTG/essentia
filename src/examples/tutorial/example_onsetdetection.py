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

from essentia import Pool, array
from essentia.standard import *


# In this example we are going to look at how to perform some onset detection
# and mark them on the audio using the AudioOnsetsMarker algorithm.
#
# Onset detection consists of two main phases:
#  1- we need to compute an onset detection function, which is a function
#     describing the evolution of some parameters, which might be representative
#     of whether we might find an onset or not
#  2- performing the actual onset detection, that is given a number of these
#     detection functions, decide where in the sound there actually are onsets


# we're going to work with a file specified as an argument in the command line
try:
    filename = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<audiofile>"
    sys.exit()

# don't forget, we can actually instantiate and call an algorithm on the same line!
print 'Loading audio file...'
audio = MonoLoader(filename = filename)()

# Phase 1: compute the onset detection function
# The OnsetDetection algorithm tells us that there are several methods available in Essentia,
# let's do two of them

od1 = OnsetDetection(method = 'hfc')
od2 = OnsetDetection(method = 'complex')

# let's also get the other algorithms we will need, and a pool to store the results

w = Windowing(type = 'hann')
fft = FFT() # this gives us a complex FFT
c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)

pool = Pool()

# let's get down to business
print 'Computing onset detection functions...'
for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    mag, phase, = c2p(fft(w(frame)))
    pool.add('features.hfc', od1(mag, phase))
    pool.add('features.complex', od2(mag, phase))


# Phase 2: compute the actual onsets locations
onsets = Onsets()

print 'Computing onset times...'
onsets_hfc = onsets(# this algo expects a matrix, not a vector
                    array([ pool['features.hfc'] ]),

                    # you need to specify weights, but as there is only a single
                    # function, it doesn't actually matter which weight you give it
                    [ 1 ])

onsets_complex = onsets(array([ pool['features.complex'] ]), [ 1 ])


# and mark them on the audio, which we'll write back to disk
# we use beeps instead of white noise to mark them, as it's more distinctive
print 'Writing audio files to disk with onsets marked...'

# mark the 'hfc' onsets:
marker = AudioOnsetsMarker(onsets = onsets_hfc, type = 'beep')
marked_audio = marker(audio)
MonoWriter(filename = 'onsets_hfc.wav')(marked_audio)

# mark the 'complex' onsets:
marker = AudioOnsetsMarker(onsets = onsets_complex, type = 'beep')
# mark the audio and make it an mp3 file, all in 1 line, just because we can!
MonoWriter(filename = 'onsets_complex.mp3', format = 'mp3')(marker(audio))


# and now go listen to your nice audio files to see which onset detection function
# works better!
print 'All done!'
print 'Results are written to the files \'onsets_hfc.wav\' and \'onsets_complex.mp3\''
