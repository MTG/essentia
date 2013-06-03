#!/usr/bin/env python

from essentia_test import *

class TestMean(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Mean(), [])

    def testZero(self):
        result = Mean()([0]*10)
        self.assertAlmostEqual(result, 0)

    def testOne(self):
        result = Mean()([100])
        self.assertAlmostEqual(result, 100)

    def testMulti(self):
        result = Mean()([5, 8, 4, 9, 1])
        self.assertAlmostEqual(result, 5.4)

    def testNegatives(self):
        result = Mean()([3, 7, -45, 2, -1, 0])
        self.assertAlmostEqual(result, -5.666666666)

    def testRational(self):
        result = Mean()([3.1459, -0.4444, .00002])
        self.assertAlmostEqual(result, 0.900506666667)


suite = allTests(TestMean)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
