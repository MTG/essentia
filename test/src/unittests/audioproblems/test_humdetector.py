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


class TestHumDetector(TestCase):
    def testZero(self):
        self.assertEqualVector(HumDetector()(esarr(np.zeros(512)))[1], esarr([]))

    def testOnes(self):
        self.assertEqualVector(HumDetector()(esarr(np.ones(512)))[1], esarr([]))

    def testInvalidParam(self):
        self.assertConfigureFails(HumDetector(), {'sampleRate': -1})
        self.assertConfigureFails(HumDetector(), {'frameSize': 0})
        self.assertConfigureFails(HumDetector(), {'hopSize': 0})
        self.assertConfigureFails(HumDetector(), {'Q0': -1})
        self.assertConfigureFails(HumDetector(), {'Q1': 2})
        self.assertConfigureFails(HumDetector(), {'minimumFrequency': 0})
        self.assertConfigureFails(HumDetector(), {'timeWindow': 0})

    def testSyntheticHum(self):
        # This test adds an artificial humming tone of 50 Hz.
        filename = join(testdata.audio_dir, 'recorded/Vivaldi_Sonata_5_II_Allegro.wav')
        audio = MonoLoader(filename=filename)()

        nSamples = len(audio)

        time = np.linspace(0, nSamples / 44100., nSamples)

        freq = 50.

        hum = np.sin(2 * np.pi * freq * time)

        hum *= db2amp(-40.)

        _, f, _, _, _ = HumDetector()(np.array(audio + hum, dtype=np.float32))

        self.assertAlmostEqualVector(f, [freq], 1e1)

    def testARProcess(self):
        # This test assess the capability of the algorithm to
        # detect a low frequency humming modeled as an autoregresive
        # process. The idea is that this kind of signal is probably
        # closer to hummings found on real scenarios.
        PI = np.pi
        time = 30.  # s
        fs = 44100.  # fs
        freq = 100  # Hz
        nSamples = int(time * fs)

        attempts = 3
        for i in range(attempts):
            try:
                noise = np.array(np.random.randn(nSamples), dtype=np.float32)

                v1 = [1, -1 * np.exp(1j * 2 * PI * freq / fs)]
                v2 = [1, -1 * np.exp(-1j * 2 * PI * freq / fs)]

                a = np.array(np.convolve(np.real(v1), np.real(v2)), dtype=np.float32)
                b = 1 - np.sum(abs(a))

                hum = IIR(denominator=a, numerator=[1.])(noise)
                hum /= np.max(hum)

                signal = hum + noise 
                signal /= np.max(signal)

                _, estimated_freqs, _, _, _, = HumDetector(detectionThreshold=2.)(signal)

                self.assertAlmostEqualVector([estimated_freqs[0]], [freq], 1e2)
            except IndexError:
                if i == attempts - 1:
                    self.assertAlmostEqualVector([estimated_freqs[0]], [freq], 1e2)
                print('testARProcess failed. This test is based on random '
                      'signals so it can fail sometimes. {}Attempt {}/{}'
                      .format('It will be repeated. ' if i < 2 else '', i+1, attempts))
                continue
            if i > 0:
                print('testARProcess passed on the attempt {}'.format(i+1))
            break

suite = allTests(TestHumDetector)


if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
