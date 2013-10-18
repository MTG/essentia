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

import sys, csv
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *

# In this script we will extract predominant melody given a music file

try:
    filename = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<input-audiofile>"
    sys.exit()




# We will use a composite algorithm PredominantMelody, which combines a number of 
# required steps for us. Let's declare and configure it first: 
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True # read the algorithm's reference for more details
run_predominant_melody = PredominantMelody(guessUnvoiced=guessUnvoiced,
                                           frameSize=frameSize,
                                           hopSize=hopSize);

# Load audio file, apply equal loudness filter, and compute predominant melody
audio = MonoLoader(filename = filename, sampleRate=sampleRate)()
audio = EqualLoudness()(audio)
pitch, confidence = run_predominant_melody(audio)


n_frames = len(pitch)
print "number of frames:", n_frames

# Visualize output pitch values
fig = plt.figure()
plot(range(n_frames), pitch, 'b')
n_ticks = 10
xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
plt.xticks(xtick_locs, xtick_lbls)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch (Hz)')
suptitle("Predominant melody pitch")

# Visualize output pitch confidence
fig = plt.figure()
plot(range(n_frames), confidence, 'b')
n_ticks = 10
xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
plt.xticks(xtick_locs, xtick_lbls)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Confidence')
suptitle("Predominant melody pitch confidence")

show()
