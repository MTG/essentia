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


import sys

import essentia.standard as es
from essentia import db2pow
from essentia import instantPower

sys.path.insert(0, './')
from qa_test import *
from qa_testevents import QaTestEvents


frameSize = 512
hopSize = 256


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.ClickDetector(frameSize=frameSize, hopSize=hopSize)

    def compute(self, *args):
        y = []
        self.algo.reset()
        for frame in es.FrameGenerator(args[1], frameSize=frameSize,
                                       hopSize=hopSize,
                                       startFromZero=True):
            starts, ends = self.algo(frame)
            if len(starts) > 0:
                for start in starts:
                    y.append(start)
        return esarr(y)


class Dev(QaWrapper):
    """
    Essentia Solution.
    """

    def compute(self, *args):
        x = args[1]
        order = 12
        LPC = es.LPC(order=order, type='regular')
        idx_ = 0
        threshold = 10
        powerEstimationThreshold = 10
        silenceThreshold = db2pow(-50)
        detectionThreshold = db2pow(30)

        start_proc = int(frameSize / 2 - hopSize / 2)
        end_proc = int(frameSize / 2 + hopSize / 2)

        y = []
        for frame in es.FrameGenerator(x, frameSize=frameSize, 
                                       hopSize=hopSize,
                                       startFromZero=True):
            if instantPower(frame) < silenceThreshold:
                idx_ += 1
                continue

            lpc, _ = LPC(frame)

            lpc /= np.max(lpc)

            e = es.IIR(numerator=lpc)(frame)

            e_mf = es.IIR(numerator=-lpc)(e[::-1])[::-1]

            # Thresholding
            th_p = np.max([self.robustPower(e, powerEstimationThreshold) *\
                           detectionThreshold, silenceThreshold])

            detections = [i + start_proc for i, v in\
                          enumerate(e_mf[start_proc:end_proc]**2) if v >= th_p]
            if detections:
                starts = [detections[0]]
                ends = []
                end = detections[0]
                for idx, d in enumerate(detections[1:], 1):
                    if d == detections[idx-1] + 1:
                        end = d
                    else:
                        ends.append(end)
                        starts.append(d)
                        end = d
                ends.append(end)

                for start in starts:
                    y.append((start + idx_ * hopSize) / 44100.)

                # for end in ends:
                #     y.append((end + idx_ * hopSize) / 44100.)

            idx_ += 1

        return esarr(y)

    def tukeyBiWeighted(self, x):
        if np.abs(x) >= 1:
            return 1/6.
        else:
            return (1-(1 - x**2)**3) / 6.

    def robustPower(self, x, k):
        robustX = np.abs(x) ** 2
        median = np.median(robustX)
        robustX[robustX > median * k] = median * k
        return np.sum(robustX) / len(robustX)

    def robustStd(self, x, k):
        robustX = np.abs(x) ** 2
        median = np.median(robustX)
        robustX[robustX > median * k] = median * k
        return np.std(robustX)

    def robustMedian(self, x, k):
        robustX = np.abs(x) ** 2
        median = np.median(robustX)
        robustX[robustX > median * k] = median * k
        return np.median(robustX)


if __name__ == '__main__':
    folder = 'clickdetector'

    # Instantiating wrappers
    wrappers = [
        EssentiaWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    data_dir = '../../QA-audio/Clicks'

    qa.load_audio(filename=data_dir, stereo=False)  # Works for a single

    qa.compute_all(output_file='{}/compute.log'.format(folder))

    # The tested signals are different version of the same file with different
    # click amplification. It was found that only clips above 9dB amplification
    # are able to be heard. This test checks that the algorithm is configured in
    # that way.
    lim = 9.  # dB
    location = 1.  # s

    def getdB(filename):
        return float(filename.split('_')[-1][:-2])

    for sol, val in qa.solutions.items():
        amplification = getdB(sol[1])
        if amplification > lim:
            if np.abs(val[0] - location) < .1:
                print('{}: {}'.format(sol, 'ok!'))
            else:
                print('{}: {}'.format(sol, 'failed'))
        else:
            if len(val) == 0:
                print('{}: {}'.format(sol, 'ok!'))
            else:
                print('{}: {}'.format(sol, 'failed'))
