#!/usr/bin/env python

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

import numpy as np

import essentia.standard as es
from essentia import array as esarr

sys.path.insert(0, './')
from qa_test import *
from qa_testevents import QaTestEvents


# parameters
sampleRate = 44100.
frameSize = 512
hopSize = 256
minimumDuration = 0.005  # ms

# inner variables
idx = 0
previousRegion = None


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.SaturationDetector(frameSize=frameSize, hopSize=hopSize)

    def compute(self, *args):
        x = args[1]
        y = []
        self.algo.reset()
        for frame in es.FrameGenerator(x, frameSize=frameSize, hopSize=hopSize,
                                       startFromZero=True):
            starts, ends = self.algo(frame)
            if len(starts) > 0:
                for start in starts:
                    y.append(start)
        return esarr(y)


class DevWrap(QaWrapper):
    """
    Development Solution.
    """
    previousRegion = None

    def compute(self, *args):
        x = args[1]
        y = []
        idx = 0
        for frame in es.FrameGenerator(x, frameSize=frameSize,
                                       hopSize=hopSize, 
                                       startFromZero=True):
            frame = np.abs(frame)
            starts = []
            ends = []

            s = int(frameSize // 2 - hopSize // 2) - 1  # consider non overlapping case
            e = int(frameSize // 2 + hopSize // 2)

            delta = np.diff(frame)
            delta = np.insert(delta, 0, 0)
            energyMask = np.array([x > .9 for x in frame])[s:e].astype(int)
            deltaMask = np.array([np.abs(x) < .01 for x in delta])[s:e].astype(int)

            combinedMask = energyMask * deltaMask

            flanks = np.diff(combinedMask)

            uFlanks = [idx for idx, x in enumerate(flanks) if x == 1]
            dFlanks = [idx for idx, x in enumerate(flanks) if x == -1]

            if self.previousRegion and dFlanks:
                start = self.previousRegion
                end = (idx * hopSize + dFlanks[0] + s) / sampleRate
                duration = start - end

                if duration > minimumDuration:
                    starts.append(start)
                    ends.append(end)

                self.previousRegion = None
                del dFlanks[0]

            if len(dFlanks) is not len(uFlanks):
                self.previousRegion = (idx * hopSize + uFlanks[-1] + s) / sampleRate
                del uFlanks[-1]

            if len(dFlanks) is not len(uFlanks):
                raise EssentiaException(
                    "Ath this point uFlanks ({}) and dFlanks ({}) "
                    "are expected to have the same length!".format(len(dFlanks),
                                                                   len(uFlanks)))

            for idx in range(len(uFlanks)):
                start = float(idx * hopSize + uFlanks[idx] + s) / sampleRate
                end = float(idx * hopSize + dFlanks[idx] + s) / sampleRate
                duration = end - start
                if duration > minimumDuration:
                    starts.append(start)
                    ends.append(end)

            for start in starts:
                y.append(start)
            idx += 1

        return esarr(y)


if __name__ == '__main__':
    folder = 'saturationdetector'

    # Instantiating wrappers
    wrappers = [
        DevWrap('events'),
        EssentiaWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    data_dir = '../../audio/recorded/distorted.wav'

    qa.load_audio(filename=data_dir)  # works for a single

    qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times.

    qa.compute_all(output_file='{}/compute.log'.format(folder))

    # TODO Generate Ground truth to test this.

    for sol, val in qa.solutions.items():
        print('{}'.format(sol))
        for v in val:
            print('{:.3f}'.format(v))
