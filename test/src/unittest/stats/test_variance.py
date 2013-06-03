#!/usr/bin/env python

from essentia_test import *

class TestVariance(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Variance(), [])

    def testZero(self):
        result = Variance()([0]*10)
        self.assertAlmostEqual(result, 0)

    def testOne(self):
        result = Variance()([100])
        self.assertAlmostEqual(result, 0)

    def testMulti(self):
        result = Variance()([5, 8, 4, 9, 1])
        self.assertAlmostEqual(result, 8.24)

    def testNegatives(self):
        result = Variance()([3, 7, -45, 2, -1, 0])
        self.assertAlmostEqual(result, 315.888889)

    def testRational(self):
        result = Variance()([3.1459, -0.4444, .00002])
        self.assertAlmostEqual(result, 2.5538138)


suite = allTests(TestVariance)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
