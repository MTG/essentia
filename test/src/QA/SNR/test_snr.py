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
from qa_testvalues import QaTestValues
import essentia.standard as es

import matplotlib.pyplot as plt
from essentia import instantPower
from essentia import db2pow

frameSize = 512
hopSize = 256


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.ClickDetector(frameSize=frameSize, hopSize=hopSize, detectionThreshold=30)

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
    Development Solution.
    """

    def compute(self, *args):

        def SNR(alpha, MMSE, noiseVar, lamb):

            lamb = lamb - 1
            lamb[lamb < 0] = 0
            return alpha * (MMSE ** 2) / noiseVar + (1 - alpha) * np.abs(lamb)

        def updateNoiseVar(pastNoiseVar, noise, alpha=.5):
            return alpha * pastNoiseVar + (1 - alpha) * np.abs(noise) ** 2

        def MMSE(v, gamma, Y):
            return G(1.5) * ( np.sqrt(v) / gamma ) * np.exp(- v / 2) * ((1 + vk) * I0(v / 2) + vk * I1(v / 2)) * Y

        x = args[1]
        idx_ = 0
        silenceThreshold = db2pow(-50)

        start_proc = int(frameSize / 2 - hopSize / 2)
        end_proc = int(frameSize / 2 + hopSize / 2)

        alpha = .5

        y = []

        pastNoiseVar = np.random.randn(frameSize / 2 + 1) * silenceThreshold
        pastNoiseVar /= np.std(pastNoiseVar)  # whiten the initial SNR value

        spectrum = es.Spectrum(size=frameSize)



        for frame in es.FrameGenerator(x, frameSize=frameSize, hopSize=hopSize, startFromZero=True):

            fft = spectrum(frame)

            if instantPower(frame) < silenceThreshold:
                idx_ += 1
                pastNoiseVar = updateNoiseVar(pastNoiseVar, fft, alpha=alpha)
            else:
                print a

            Yvar =  fft **2

            SNRprior = Yvar / pastNoiseVar

            # plt.plot(pastNoiseVar)
            # plt.plot(Yvar)
            # plt.show()

            plt.close()

            idx_ += 1

        return esarr(y)


if __name__ == '__main__':
    folder = 'SNR'

    # Instantiating wrappers
    wrappers = [
        Dev('events'),
    ]

    # Instantiating the test
    qa = QaTestValues(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/Vivaldi_Sonata_5_II_Allegro_prominent_jump.wav'
    # data_dir = '../../QA-audio/Clicks'
    # data_dir = '/home/pablo/reps/essentia/test/audio/recorded/vignesh.wav'
    data_dir = '/home/pablo/reps/essentia/test/audio/recorded/mozart_c_major_30sec.wav'
    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/'
    # data_dir = '../../QA-audio/Discontinuities/prominent_jumps/vignesh_prominent_jump.wav'
    # data_dir = '../../../../../../pablo/Music/Desakato-La_Teoria_del_Fuego/'

    qa.load_audio(filename=data_dir, stereo=False)  # Works for a single
    # qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times
    qa.compute_all(output_file='{}/compute.log'.format(folder))

    # qa.score_all()
    # qa.scores
    # qa.save_test('{}/test'.format(folder))
