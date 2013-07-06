#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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



from essentia_test import *
from essentia import *
from numpy import random

class TestSilenceRate(TestCase):

    def evaluateSilenceRate(self, input):
        thresh = [0, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 0.8]
        nThresh = len(thresh)
        nFrames = len(input)

        # expected values:
        expected = zeros([nFrames, nThresh])
        for frame in range(nFrames):
            if len(input[frame]):
                power = instantPower(input[frame])
            for i in range(nThresh):
                if power < thresh[i]: expected[frame][i] = 1
                else: expected[frame][i] = 0

        output = []
        for frame in input:
          output.append(SilenceRate(thresholds = thresh)(frame))
        self.assertAlmostEqualMatrix(output, expected)


    def testRegression(self):
        size = 100
        nFrames = 10
        input = zeros([nFrames, size])
        for i in range(nFrames):
            for j in range(size):
                input[i][j] = random.rand()*2.0-1.0
        self.evaluateSilenceRate(input)

    def testOnes(self):
        size = 100
        nFrames = 10
        input = ones([nFrames, size])
        self.evaluateSilenceRate(input)

    def testZeros(self):
        size = 100
        nFrames = 10
        input = zeros([nFrames, size])
        self.evaluateSilenceRate(input)

    def testEmpty(self):
        self.assertComputeFails(SilenceRate(thresholds = [0.1]), [])



suite = allTests(TestSilenceRate)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
