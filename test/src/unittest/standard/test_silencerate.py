#!/usr/bin/env python

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
