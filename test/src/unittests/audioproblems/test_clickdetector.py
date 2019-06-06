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
        self.assertEqualVector(ClickDetector()(esarr(np.zeros(512)))[0],
                               esarr([]))

    def testOnes(self):
        self.assertEqualVector(ClickDetector()(esarr(np.ones(512)))[0],
                               esarr([]))

    def testInvalidParam(self):
        self.assertConfigureFails(ClickDetector(), {'sampleRate': -1})
        self.assertConfigureFails(ClickDetector(), {'frameSize': 0})
        self.assertConfigureFails(ClickDetector(), {'hopSize': 0})
        self.assertConfigureFails(ClickDetector(), {'order': 0})
        self.assertConfigureFails(ClickDetector(), {'order': 5, 'frameSize': 4})
        self.assertConfigureFails(ClickDetector(), {'powerEstimationThreshold': 0})
        self.assertConfigureFails(ClickDetector(), {'silenceThreshold': 1})

    def testRegression(self, frameSize=512, hopSize=256):
        fs = 44100.

        audio = MonoLoader(filename=join(testdata.audio_dir,
                           'recorded/vignesh.wav'),
                           sampleRate=fs)()

        originalLen = len(audio)
        jumpLocation1 = int(originalLen / 4.)
        jumpLocation2 = int(originalLen / 2.)
        jumpLocation3 = int(originalLen * 3 / 4.)

        audio[jumpLocation1] += .1
        audio[jumpLocation2] += .08
        audio[jumpLocation3] += .05

        groundTruth = esarr([jumpLocation1, jumpLocation2, jumpLocation3]) / fs

        clickStarts = []
        clickEnds = []
        clickdetector = ClickDetector(frameSize=frameSize, hopSize=hopSize)

        for frame in FrameGenerator(audio, frameSize=frameSize,
                                    hopSize=hopSize, startFromZero=True):
            starts, ends = clickdetector(frame)
            if not len(starts) == 0:
                for start in starts:
                    clickStarts.append(start)

                for end in ends:
                    clickEnds.append(end)

        self.assertAlmostEqualVector(clickStarts, groundTruth, 1e-5)

        self.assertAlmostEqualVector(clickEnds, groundTruth, 1e-5)

    def testMinimumOverlap(self):
        self.testRegression(frameSize=512, hopSize=488)

suite = allTests(TestClickDetector)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
