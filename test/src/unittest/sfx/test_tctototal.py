#!/usr/bin/env python

from essentia_test import *
from math import exp

class TestTCToTotal(TestCase):

    def testEmpty(self):
        self.assertComputeFails(TCToTotal(), [])

    def testOne(self):
        self.assertComputeFails(TCToTotal(), [])

    def testImpulseBeginning(self):
        self.assertAlmostEqual(TCToTotal()([1,0]), 0)

    def testImpulseMiddle(self):
        self.assertAlmostEqual(TCToTotal()([0,1,0]), 0.5)

    def testTriangle(self):
        size = 100
        envelope = zeros(size)

        for i in range(size/2):
            envelope[i] = i
        for i in range(size/2, size):
            envelope[i] = size - i
        TCToTotal()(envelope)
        self.assertAlmostEqual(TCToTotal()(envelope), 0.5*size/float(size-1))

    def testImpulseEnd(self):
        self.assertAlmostEqual(TCToTotal()([0,1]), 1)

    def testFlat(self):
        self.assertAlmostEqual(TCToTotal()([1]*100), 0.5)

    def testZero(self):
        self.assertComputeFails(TCToTotal(), [0]*100)

    def testGaussian(self):
        data = [x/100. for x in xrange(-50, 50)]
        envelope = [exp(-(x**2)/2) for x in data]
        self.assertAlmostEqual(TCToTotal()(envelope), 0.5, 1e-3)

    def testAlternating(self):
        self.assertComputeFails(TCToTotal(), [1,-1,1,-1,1,-1,1,-1])


suite = allTests(TestTCToTotal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
