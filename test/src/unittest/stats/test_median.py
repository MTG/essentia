#!/usr/bin/env python

from essentia_test import *

class TestMedian(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Median(), [])

    def testZero(self):
        result = Median()([0]*10)
        self.assertEqual(result, 0)

    def testOne(self):
        result = Median()([100])
        self.assertEqual(result, 100)

    def testMulti(self):
        result = Median()([5, 8, 4, 9, 1])
        self.assertEqual(result, 5)

    def testNegatives(self):
        result = Median()([3, 7, -45, 2, -1, 0])
        self.assertEqual(result, 1)

    def testRational(self):
        result = Median()([3.1459, -0.4444, .00002])
        self.assertAlmostEqual(result, 0.00002)

    def testEvenSize(self):
        result = Median()([1, 4, 3, 10])
        self.assertAlmostEqual(result, 3.5)


suite = allTests(TestMedian)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
