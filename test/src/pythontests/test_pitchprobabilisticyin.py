#!/usr/bin/env python

# Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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
import essentia.streaming as ess
import librosa
import time
import os

filename = os.path.join(os.path.dirname(__file__) ,'../../audio/recorded', 'long_voice.wav')
loader = ess.MonoLoader(filename=filename, sampleRate=48000)
PYIN = ess.PitchYinProbabilistic(sampleRate=48000, frameSize=2048, hopSize=256, lowRMSThreshold=0.1, outputUnvoiced=2)
start_time = time.time()
loader.audio >> PYIN.signal
pool = essentia.Pool()
PYIN.pitch >> (pool, 'pitch')
PYIN.voicedProbabilities >> (pool, 'voicedProbabilities')
essentia.run(loader)
end_time = time.time()
print("Execution time {} s".format(end_time-start_time))
