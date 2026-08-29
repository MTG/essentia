# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
import essentia.standard as es
from pylab import *
from numpy import *

# In this script we will extract predominant melody given a music file

try:
    filename = sys.argv[1]
except:
    print(f"usage: {sys.argv[0]} <input-audiofile>")
    sys.exit()


# We will use a composite algorithm PredominantMelody, which combines a number of
# required steps for us. Let's declare and configure it first:
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True  # read the algorithm's reference for more details
run_predominant_melody = es.PitchMelodia(
    guessUnvoiced=guessUnvoiced, frameSize=frameSize, hopSize=hopSize
)

# Load audio file, apply equal loudness filter, and compute predominant melody
audio = es.MonoLoader(filename=filename, sampleRate=sampleRate)()
audio = es.EqualLoudness()(audio)
pitch, confidence = run_predominant_melody(audio)


n_frames = len(pitch)
print(f"number of frames: {n_frames}")

# Visualize output pitch values
fig, ax = plt.subplots(1, figsize=(10, 4))
ax.plot(range(n_frames), pitch, "b")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pitch (Hz)")
ax.set_xlim([0, n_frames - 1])

n_ticks = 10
xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
xtick_lbls = [
    i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)
]
xtick_lbls = [f"{round(x, 2):.2f}" for x in xtick_lbls]

plt.sca(ax)
plt.xticks(xtick_locs, xtick_lbls)

suptitle("Predominant melody pitch")
tight_layout()
show()

# Visualize output pitch confidence
fig, ax = plt.subplots(1, figsize=(10, 4))
ax.plot(range(n_frames), confidence, "b")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Confidence")
ax.set_xlim([0, n_frames - 1])

plt.sca(ax)
plt.xticks(xtick_locs, xtick_lbls)

suptitle("Predominant melody pitch confidence")
tight_layout()
show()
