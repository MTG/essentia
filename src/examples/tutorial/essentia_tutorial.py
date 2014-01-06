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

"""Demo of Essentia 'standard' mode.

This first demo will show how to use Essentia in standard mode.
This will require a little bit of knowledge of python (not that much!) and
will look like an interactive session in matlab.

We will have a look at some basic functionality:
 - how to load an audio
 - how to perform some numerical operations, such as FFT et al.
 - how to plot results
 - how to output results to a file

To run this demo interactively, open IPython and type in the following commands:
    from IPython.lib.demo import Demo
    essentia_demo = Demo('essentia_tutorial.py')

Type command
    essentia_demo()
to show and execute each block of the demo. Each block of code will be printed to 
the screen before it is run. This is another nifty feature of the IPython 
interpreter. As we go along the demo, we will also be looking at a few IPython 
features that make your life easier.

So, let's start!
"""
# <demo> --- stop ---


# first, we need to import our essentia module. It is aptly named 'essentia'!

import essentia

# as there are 2 operating modes in essentia which have the same algorithms,
# these latter are dispatched into 2 submodules:

import essentia.standard
import essentia.streaming

# let's have a look at what's in there

print dir(essentia.standard)


# <demo> --- stop ---


# let's define a small utility function

def play(audiofile):
    import os, sys

    # NB: this only works with linux!! mplayer rocks!
    if sys.platform == 'linux2':
        os.system('mplayer %s' % audiofile)
    else:
        print 'Not playing audio...'

# So, first things first, let's load an audio
# to make sure it's not a trick, let's show the original "audio" to you:

play('../../../test/audio/recorded/dubstep.wav')


# <demo> --- stop ---


# Essentia has a selection of audio loaders:
#
#  - AudioLoader: the basic one, returns the audio samples, sampling rate and number of channels
#  - MonoLoader: which returns audio, down-mixed and resampled to a given sampling rate
#  - EasyLoader: a MonoLoader which can optionally trim start/end slices and rescale according
#                to a ReplayGain value
#  - EqloudLoader: an EasyLoader that applies an equal-loudness filtering on the audio
#

# we start by instantiating the audio loader:
loader = essentia.standard.MonoLoader(filename = '../../../test/audio/recorded/dubstep.wav')

# and then we actually perform the loading:
audio = loader()


# <demo> --- stop ---


# OK, let's make sure the loading process actually worked

from pylab import *

plot(audio[1*44100:2*44100])
show()

# <demo> --- stop ---


# So, let's get down to business:
# Let's say we want to analyze the audio frame by frame, and we want to compute
# the MFCC for each frame. We will need the following algorithms:
# Windowing, FFT, MFCC

from essentia.standard import *
w = Windowing(type = 'hann')
spectrum = Spectrum()  # FFT() would give the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()

help(MFCC)

# <demo> --- stop ---

# once algorithms have been instantiated, they work like functions:
frame = audio[5*44100 : 5*44100 + 1024]
spec = spectrum(w(frame))

plot(spec)
show()

# <demo> --- stop ---

# let's try to compute the MFCCs for all the frames in the audio:

mfccs = []
frameSize = 1024
hopSize = 512

for fstart in range(0, len(audio)-frameSize, hopSize):
    frame = audio[fstart:fstart+frameSize]
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)

# and plot them...
# as this is a 2D array, we need to use imshow() instead of plot()
imshow(mfccs, aspect = 'auto')
show()


# <demo> --- stop ---

# and let's do it in a more essentia-like way:

mfccs = []

for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)

# transpose to have it in a better shape
mfccs = essentia.array(mfccs).T

imshow(mfccs[1:,:], aspect = 'auto')
show()

# <demo> --- stop ---

# Introducing the Pool: a good-for-all container
#
# A Pool can contain any type of values (easy in Python, not as much in C++ :-) )
# They need to be given a name, which represent the full path to these values;
# dot '.' characters are used as separators. You can think of it as a directory
# tree, or as namespace(s) + local name.
#
# Examples of valid names are: bpm, lowlevel.mfcc, highlevel.genre.rock.probability, etc...

# So let's redo the previous using a Pool

pool = essentia.Pool()

for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    pool.add('lowlevel.mfcc_bands', mfcc_bands)

imshow(pool['lowlevel.mfcc'].T[1:,:], aspect = 'auto')
figure()
# Let's plot mfcc bands on a log-scale so that the energy values will be better 
# differentiated by color
from matplotlib.colors import LogNorm
imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', interpolation = 'nearest', norm = LogNorm())
show()


# <demo> --- stop ---

# In essentia there is mostly 1 way to output your data in a file: the YamlOutput
# although, as all of this is done in python, it should be pretty easy to output to
# any type of data format.

output = YamlOutput(filename = 'mfcc.sig')
output(pool)

# <demo> --- stop ---

# Say we're not interested in all the MFCC frames, but just their mean & variance.
# To this end, we have the PoolAggregator algorithm, that can do all sorts of
# aggregation: mean, variance, min, max, etc...

aggrPool = PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

print 'Original pool descriptor names:'
print pool.descriptorNames()
print
print 'Aggregated pool descriptor names:'
print aggrPool.descriptorNames()

output = YamlOutput(filename = 'mfccaggr.sig')
output(aggrPool)

