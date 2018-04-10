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
import essentia.standard as es

import matplotlib.pyplot as plt
from essentia import instantPower
from essentia import db2pow

frameSize = 512
hopSize = 450


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.ClickDetector(frameSize=frameSize, hopSize=hopSize)

    def compute(self, *args):
        y = []
        self.algo.reset()
        for frame in es.FrameGenerator(args[1], frameSize=frameSize, hopSize=hopSize,
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
        for frame in es.FrameGenerator(x, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
            if instantPower(frame) < silenceThreshold:
                idx_ += 1
                continue

            lpc, _ = LPC(frame)

            lpc /= np.max(lpc)

            e = es.IIR(numerator=lpc)(frame)

            e_mf = es.IIR(numerator=-lpc)(e[::-1])[::-1]

            # e[:order] = np.zeros(order)
            # e_mf[:order] = np.zeros(order)

            # Thresholding
            th_p = np.max([self.robustPower(e, powerEstimationThreshold) * detectionThreshold, silenceThreshold])

            # plt.plot(frame)
            # plt.plot(e[order:] ** 2, label='e')
            # plt.plot(e_mf[order:] ** 2, label='mf')
            # #
            # plt.axhline(th_p, color='r', label='th power')
            # plt.legend()
            # plt.show()
            # plt.close()

            detections = [i + start_proc for i, v in enumerate(e_mf[start_proc:end_proc]**2) if v >= th_p]
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
    folder = 'ClickDetector'

    # We are using 1 digit only to fit the format of PyloudnessWrap
    # np.set_printoptions(precision=1)

    # Instantiating wrappers
    wrappers = [
        EssentiaWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/Vivaldi_Sonata_5_II_Allegro_prominent_jump.wav'
    data_dir = '../../QA-audio/Discontinuities/prominent_jumps/'
    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/vignesh_prominent_jump.wav'
    #  data_dir = '../../../../../../pablo/Music/Desakato-La_Teoria_del_Fuego/'

    qa.load_audio(filename=data_dir, stereo=False)  # Works for a single
    qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times
    qa.compute_all(output_file='{}/compute.log'.format(folder))

    qa.score_all()
    # qa.scores
    # qa.save_test('{}/test'.format(folder))

    x = qa.data['vignesh_prominent_jump'][33863:33863+512]
    # Add the testing files
