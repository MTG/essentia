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


import numpy as np
from math import *

from essentia_test import *
from essentia import array as esarr


class TestClickDetector(TestCase):
    def testZero(self):
        self.assertEqual(SNR()(esarr(np.zeros(512)))[1], -np.inf)

    def testOnes(self):
        self.assertEqual(SNR()(esarr(np.ones(512)))[1], np.inf)

    def testInvalidParam(self):
        self.assertConfigureFails(SNR(), {'sampleRate': -1})
        self.assertConfigureFails(SNR(), {'frameSize': 0})
        self.assertConfigureFails(SNR(), {'noiseThreshold': 1})
        self.assertConfigureFails(SNR(), {'MMSEAlpha': 2})
        self.assertConfigureFails(SNR(), {'MAAlpha': 2})
        self.assertConfigureFails(SNR(), {'NoiseAlpha': 2})

    def testSinusoidalPlusNoise(self):
        from essentia import instantPower
        from essentia import db2amp
        frameSize = 512
        hopSize = frameSize // 2
        fs = 44100.
        time = 5.  # s
        time_axis = np.arange(0, time, 1 / fs)
        nsamples = len(time_axis)
        noise = np.random.randn(nsamples)
        noise /= np.std(noise)
        noise_only = 1

        signal = np.sin(2 * pi * 5000 * time_axis)

        signal_db = -22.
        noise_db  = -50.

        signal[:int(noise_only * fs)] = np.zeros(int(noise_only * fs))
        snr_gt = 10. * np.log10(
            (instantPower(esarr(db2amp(signal_db) * signal[int(noise_only * fs):]))) /
            (instantPower(esarr(db2amp(noise_db)  * noise[int(noise_only * fs):]))))\
            - 10. * np.log10(fs / 2.)

        signal_and_noise = esarr(db2amp(signal_db) * signal + db2amp(noise_db) * noise)

        noiseThreshold = -30
        algo = SNR(frameSize=frameSize, noiseThreshold=noiseThreshold)
        for frame in FrameGenerator(signal_and_noise, frameSize=frameSize, hopSize=hopSize):
            _, snr, _ = algo(frame)

        self.assertAlmostEqual(snr, snr_gt, 1e-1)

    def testBroadbandNoiseCorrection(self):
        from essentia import instantPower
        from essentia import db2amp
        frameSize = 512
        hopSize = frameSize // 2
        fs = 44100.
        time = 1.  # s
        time_axis = np.arange(0, time, 1 / fs)
        nsamples = len(time_axis)
        noise = np.random.randn(nsamples)
        noise /= np.std(noise)
        noise_only = .2

        signal = np.sin(2 * pi * 5000 * time_axis)

        signal_db = -22.
        noise_db = -50.

        signal[:int(noise_only * fs)] = np.zeros(int(noise_only * fs))

        signal_and_noise = esarr(db2amp(signal_db) * signal + db2amp(noise_db) * noise)

        noiseThreshold = -30
        corrected = SNR(frameSize=frameSize, noiseThreshold=noiseThreshold)
        notCorrected = SNR(frameSize=frameSize, noiseThreshold=noiseThreshold,
                           useBroadbadNoiseCorrection=False)

        for frame in FrameGenerator(signal_and_noise, frameSize=frameSize, hopSize=hopSize):
            _, snrCorrected, _ = corrected(frame)
            _, snrNotCorrected, _ = notCorrected(frame)

        self.assertAlmostEqual(snrCorrected, snrNotCorrected - 10. * np.log10(fs / 2), 1e-4)

suite = allTests(TestClickDetector)


if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
