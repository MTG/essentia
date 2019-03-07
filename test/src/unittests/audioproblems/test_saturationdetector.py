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


class TestSaturationDetector(TestCase):
    def testZero(self):
        self.assertEqualVector(SaturationDetector()(esarr(np.zeros(512)))[0],
                               esarr([]))

    def testOnes(self):
        input = esarr(np.ones(512))
        input[0] = 0
        input[-1] = 0

        self.assertAlmostEqualVector(SaturationDetector(hopSize=512)(input)[0],
                                     esarr([0.]), 1e-4)

    def testLongSaturation(self, frameSize=512, hopSize=256):
        fs = 44100

        signal = [0]*fs + [1]*fs + [0]*fs

        starts = []
        ends = []
        sd = SaturationDetector(frameSize=frameSize, hopSize=hopSize)
        for frame in FrameGenerator(esarr(signal), frameSize=frameSize,
                                    hopSize=hopSize, startFromZero=True):
            s, e = sd(frame)
            starts += list(s)
            ends += list(e)

        self.assertAlmostEqualVector(starts, [1.], 1e-4)
        self.assertAlmostEqualVector(ends, [2.], 1e-4)

    def testNoOverlap(self):
        self.testLongSaturation(frameSize=512, hopSize=512)

    def testSquareWaves(self):
        # The algorithm should be able to detect the positive part
        # of square waves at different frequencies.

        fs = 44100
        minFreq = 100  # Hz
        maxFreq = 10000  # Hz
        time = .1  # s

        sd = SaturationDetector(minimumDuration=.0, hopSize=512)
        for f in numpy.linspace(minFreq, maxFreq, 5):
            sampleNum = int(fs / f)
            sampleNum -= sampleNum % 2

            waveTable = [0] * sampleNum
            waveTable[:sampleNum // 2] = [1] * (sampleNum // 2)

            waveDur = len(waveTable) / 44100.
            repetitions = int(time / waveDur)

            realStarts = esarr(range(0, repetitions)) * waveDur
            realEnds = realStarts + waveDur

            input = waveTable * repetitions

            starts = esarr([])
            ends = esarr([])
            for frame in FrameGenerator(input, frameSize=512,
                                        hopSize=512, startFromZero=True):
                s, e = sd(frame)

                starts = np.hstack([starts, s])
                ends = np.hstack([ends, e])

            self.assertAlmostEqualVectorFixedPrecision(starts, realStarts, 2)
            self.assertAlmostEqualVectorFixedPrecision(ends, realEnds, 2)
            sd.reset()

    def testInvalidParam(self):
        self.assertConfigureFails(SaturationDetector(), {'sampleRate': -1})
        self.assertConfigureFails(SaturationDetector(), {'frameSize': 0})
        self.assertConfigureFails(SaturationDetector(), {'hopSize': 1024,
                                                        'frameSize': 512})
        self.assertConfigureFails(SaturationDetector(), {'energyThreshold': 1})
        self.assertConfigureFails(SaturationDetector(), {'differentialThreshold': -1})
        self.assertConfigureFails(SaturationDetector(), {'minimumDuration': -1})

suite = allTests(TestSaturationDetector)


if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
