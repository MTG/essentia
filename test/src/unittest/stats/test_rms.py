#!/usr/bin/env python

from essentia_test import *

class TestRMS(TestCase):

    def testEmpty(self):
        self.assertComputeFails(RMS(), [])

    def testZero(self):
        result = RMS()([0]*10)
        self.assertAlmostEqual(result, 0)

    def testRegression(self):
        result = RMS()([3, 7.32, -45, 2, -1.453, 0])
        self.assertAlmostEqual(result, 18.680174914420192)

    def testSine(self):
        from numpy import sin, sqrt, pi
        size = 1000
        sine = [sin(440.0*2.0*pi*i/size) for i in range(size)]
        self.assertAlmostEqual(RMS()(sine), 1.0/sqrt(2.0), 1e-6)


suite = allTests(TestRMS)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
