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


class TestTruePeakDetector(TestCase):
    def testZero(self):
        self.assertEqualVector(TruePeakDetector()(esarr(np.zeros(512)))[0],
                               esarr([]))

    def testSinc(self):
        # This test asserts that the estimated peak position is better than
        # the sampled one. This test is performed over a sinc wave sampled
        # with different offsets.
        duration = 10  # s
        fs = 1  # hz
        k = 1.5  # amplitude
        oversamplingFactor = 4  # factor of oversampling for the real signal
        nSamples = fs * duration

        time = np.arange(-nSamples/2, nSamples/2,
                         2 ** -oversamplingFactor, dtype='float')
        samplingPoints = time[::2 ** oversamplingFactor]

        def shifted_sinc(x, k, offset):
            xShifted = x - offset
            y = np.zeros(len(xShifted))
            for idx, i in enumerate(xShifted):
                if not i:
                    y[idx] = k
                else:
                    y[idx] = (k * np.sin(np.pi * i) / (np.pi * i))
            return y

        sampledError = 0
        estimatedError = 0
        its = 10
        for offset in np.linspace(0, 1, its):
            yReal = shifted_sinc(time, k, offset)
            realPeak = np.max(yReal)

            y = shifted_sinc(samplingPoints, k, offset)
            sampledPeak = np.max(y)
            sampledError += np.abs(sampledPeak - realPeak)

            _, processed = TruePeakDetector(version=2)(y.astype(np.float32))
            estimatedPeak = np.max(processed)
            estimatedError += np.abs(estimatedPeak - realPeak)
        sampledError /= float(its)
        estimatedError /= float(its)

        # Check that the peak stimation error is reduced.
        assert(estimatedError < sampledError)

    def testInvalidParam(self):
        self.assertConfigureFails(TruePeakDetector(), {'sampleRate': -1})
        self.assertConfigureFails(TruePeakDetector(), {'oversamplingFactor': 0})
        self.assertConfigureFails(TruePeakDetector(), {'quality': 5})

    def testDifferentBitDepths(self):
        audio16 = MonoLoader(filename=join(testdata.audio_dir, 'recorded/cat_purrrr.wav'),
                           sampleRate=44100)()
        audio24 = MonoLoader(filename=join(testdata.audio_dir, 'recorded/cat_purrrr24bit.wav'),
                             sampleRate=44100)()
        audio32 = MonoLoader(filename=join(testdata.audio_dir, 'recorded/cat_purrrr32bit.wav'),
                             sampleRate=44100)()

        data = [audio16, audio24, audio32]

        peakDetector = TruePeakDetector()
        results = [peakDetector(x)[0] for x in data]

        # The algorithm should detect the same frames regardless the bit size.
        for version in results[:-1]:
            self.assertEqualVector(results[-1], version)

    def testDCblock(self):
        # This negative peak is hidden by a huge amount of positive dc offset.
        # It should be detected when the optional dc blocker is on.
        oversamplingFactor = 4
        signal = np.zeros(512)
        peakLoc = 256
        signal[peakLoc] = -1.4
        signalWithDC = signal + .5

        withoutDC = TruePeakDetector(blockDC=True,
                                     oversamplingFactor=oversamplingFactor,
                                     version=2)(signalWithDC.astype(np.float32))[0]
        withDC = TruePeakDetector(blockDC=False,
                                  oversamplingFactor=oversamplingFactor,
                                  version=2)(signalWithDC.astype(np.float32))[0]

        assert(withDC.size == 0)
        assert(peakLoc in withoutDC)


suite = allTests(TestTruePeakDetector)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
