#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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


from qa_test import *
from qa_testevents import QaTestEvents
import numpy as np
from librosa.effects import trim
import essentia.standard as es
from essentia import array as esarr
from scipy.signal import medfilt

order = 3
frame_size = 512
hop_size = 256
kernel_size = 7
times_thld = 8
energy_thld = 0.001
sub_frame = 32

class DevWrap(QaWrapper):
    """
    Development Solution.
    """

    # parameters
    _sampleRate = 44100.
    _frameSize = 512
    _hopSize = 256
    _minimumDuration = 0  # ms

    _minimumDuration /= 1000.

    # inner variables
    _idx = 0
    _previousRegion = None

    def compute(self, x):
        y = []
        self._idx = 0
        for frame in es.FrameGenerator(x, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
            frame = np.abs(frame)
            starts = []
            ends = []

            s = int(self._frameSize / 2 - self._hopSize / 2) - 1  # consider non overlapping case
            e = int(self._frameSize / 2 + self._hopSize / 2)

            delta = np.diff(frame)
            delta = np.insert(delta, 0, 0)
            energyMask = np.array([x > .9 for x in frame])[s:e].astype(int)
            deltaMask = np.array([np.abs(x) < .01 for x in delta])[s:e].astype(int)

            combinedMask = energyMask * deltaMask

            flanks = np.diff(combinedMask)
            #np.insert(delta, 0, 0)

            uFlanks = [idx for idx, x in enumerate(flanks) if x == 1]
            dFlanks = [idx for idx, x in enumerate(flanks) if x == -1]

            if self._previousRegion and dFlanks:
                start = self._previousRegion
                end = (self._idx * hop_size + dFlanks[0] + s) / self._sampleRate
                duration = start - end

                if duration > self._minimumDuration:
                    starts.append(start)
                    ends.append(end)

                self._previousRegion = None
                del dFlanks[0]

            if len(dFlanks) is not len(uFlanks):
                self._previousRegion = (self._idx * hop_size + uFlanks[-1] + s) / self._sampleRate
                del uFlanks[-1]

            if len(dFlanks) is not len(uFlanks):
                raise EssentiaException(
                    "Ath this point uFlanks ({}) and dFlanks ({}) are expected to have the same length!".format(len(dFlanks),
                                                                                                            len(uFlanks)))

            for idx in range(len(uFlanks)):
                start = float(self._idx * hop_size + uFlanks[idx] + s) / self._sampleRate
                end = float(self._idx * hop_size + dFlanks[idx] + s) / self._sampleRate
                duration = end - start
                if duration > self._minimumDuration:
                    starts.append(start)
                    ends.append(end)

            for start in starts:
                y.append(start)
            self._idx += 1

        return esarr(y)


if __name__ == '__main__':

    # Instantiating wrappers
    wrappers = [
        DevWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    data_dir = '../../audio/recorded/distorted.wav'

    qa.load_audio(filename=data_dir)  # Works for a single
    qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times

    qa.compute_all()

    # qa.score_all()

