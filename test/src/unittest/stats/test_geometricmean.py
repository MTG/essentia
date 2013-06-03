#!/usr/bin/env python

from essentia_test import *

class TestGeometricMean(TestCase):

    def testZero(self):
        self.assertEqual(GeometricMean()(zeros(1000)), 0)

    def testEmpty(self):
        self.assertComputeFails(GeometricMean(), [])

    def testNullValue(self):
        self.assertEqual(GeometricMean()([1, 23, 46, 2, 6, 13, 0, 35, 57, 3]), 0)

    def testInvalidInput(self):
        self.assertComputeFails(GeometricMean(), [ 1, 5, 3, -7, 2, 67 ])

    def testRegression(self):
        self.assertAlmostEqual(GeometricMean()([2]), 2)
        self.assertAlmostEqual(GeometricMean()([0.25, 0.5, 1]), 0.5)
        self.assertAlmostEqual(GeometricMean()([32, 16, 2, 4, 8]), 8, 2e-7)



suite = allTests(TestGeometricMean)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

