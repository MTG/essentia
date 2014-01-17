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

"""Demo of Essentia 'streaming' mode.

To run this demo interactively, open IPython and type in the following commands:
    from IPython.lib.demo import Demo
    essentia_demo = Demo('essentia_tutorial.py')

Type command
    essentia_demo()
to show and execute each block of the demo.

This second demo will show how to use Essentia in streaming mode.

The main difference between standard and streaming is that the standard mode
is imperative while the streaming mode is declarative. That means that in standard
mode, you tell exactly the computer what to do, whereas in the streaming mode, you
"declare"  what is needed to be done, and you let the computer do it itself.

One big advantage of the streaming mode is that the memory consumption is greatly
reduced, as you don't need to load the entire audio in memory.

Let's have a look at it.
"""
# <demo> --- stop ---


# as usual, first import the essentia module
import essentia
from essentia.streaming import *

# and instantiate our algorithms

loader = MonoLoader(filename = '../../../test/audio/recorded/dubstep.wav')
frameCutter = FrameCutter(frameSize = 1024, hopSize = 512)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC()

# <demo> --- stop ---

# In streaming, instead of calling algorithms like functions, we need to
# connect their inputs and outputs. This is done using the >> operator
#
# The graph we want to connect looks like this:
#  ____________       _________________       ________________       __________________
# /            \     /                 \     /                \     /                  \
# | MonoLoader |     |   FrameCutter   |     |   Windowing    |     |     Spectrum     |
# |            |     |                 |     |                |     |                  |
# |     audio -+-->--+- signal  frame -+-->--+- frame  frame -+-->--+- frame spectrum -+-->-\
# \____________/     \_________________/     \________________/     \__________________/    |
#                                                                                           |
#  /----------------------------------------------------------------------------------------/
#  |    ___________________
#  |   /                   \
#  |   |      MFCC         |
#  |   |            bands -+-->-- ???
#  \->-+- spectrum         |
#      |             mfcc -+-->-- ???
#      \___________________/
#

loader.audio >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum >> mfcc.spectrum

# <demo> --- stop ---

# When building a network, all inputs need to be connected, no matter what, otherwise the network
# cannot be started

essentia.run(loader)

# <demo> --- stop ---

#  ____________       _________________       ________________       __________________
# /            \     /                 \     /                \     /                  \
# | MonoLoader |     |   FrameCutter   |     |   Windowing    |     |     Spectrum     |
# |            |     |                 |     |                |     |                  |
# |     audio -+-->--+- signal  frame -+-->--+- frame  frame -+-->--+- frame spectrum -+-->-\
# \____________/     \_________________/     \________________/     \__________________/    |
#                                                                                           |
#  /----------------------------------------------------------------------------------------/
#  |    ___________________              _________
#  |   /                   \            /         \
#  |   |      MFCC         |      /-->--+ NOWHERE |
#  |   |            bands -+-->--/      \_________/
#  \->-+- spectrum         |
#      |             mfcc -+-->--\       _____________________
#      \___________________/      \     /                     \
#                                  -->--+ Pool: lowlevel.mfcc |
#                                       \_____________________/

pool = essentia.Pool()

mfcc.bands >> None
mfcc.mfcc >> (pool, 'lowlevel.mfcc')

essentia.run(loader)

print 'Pool contains %d frames of MFCCs' % len(pool['lowlevel.mfcc'])

# <demo> --- stop ---

# Let's try writing directly to a text file, no pool and no yaml files

# we first need to disconnect the old connection to the pool to avoid putting the same
# data in there again
mfcc.mfcc.disconnect((pool, 'lowlevel.mfcc'))

# we create a FileOutput
fileout = FileOutput(filename = 'mfccframes.txt')

# and connect it: it is a special connection as it has no input, because it can actually
# take any type of input (the other algorithms will complain if you try to connect an output
# to an input of a different type)
mfcc.mfcc >> fileout

# reset the network otherwise the loader in particular will not do anything useful
essentia.reset(loader)

# and rerun it!
essentia.run(loader)


