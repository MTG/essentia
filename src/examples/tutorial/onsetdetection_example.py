#!/usr/bin/env python

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


# we're going to work with this small loop downloaded from freesound

# don't forget, we can actually instantiate and call an algorithm on the same line!
print 'Loading audio file...'
audio = MonoLoader(filename = '554__bebeto__Ambient_loop.mp3')()

# Phase 1: compute the onset detection function
# The OnsetDetection algorithm tells us that there are 2 available in Essentia,
# so let's do both of them

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

marker = AudioOnsetsMarker(onsets = onsets_hfc, type = 'beep')
marked_audio = marker(audio)

MonoWriter(filename = 'onsets_hfc.wav')(marked_audio)

# mark the 'complex' onsets
marker = AudioOnsetsMarker(onsets = onsets_complex, type = 'beep')

# mark the audio and make it an mp3 file, all in 1 line, just because we can!
MonoWriter(filename = 'onsets_complex.mp3', format = 'mp3')(marker(audio))


# and now go listen to your nice audio files to see which onset detection function
# works better!
print 'All done!'
