#!/usr/bin/env python

from essentia_test import *
import numpy

class TestFlatnessSfx(TestCase):

    def testEmpty(self):
        self.assertComputeFails(FlatnessSFX(), [])

    def testZero(self):
        self.assertEqual(FlatnessSFX()([0]*100), 1.0)

    def testOne(self):
        self.assertEqual(FlatnessSFX()([1234]), 1.0)

    def testFlat(self):
        self.assertEqual(FlatnessSFX()([0.5]*100), 1.0)

    def testSteep(self):
        self.assertEqual(FlatnessSFX()([0, 1]) > 1.0, True)

    def testRegression(self):
        self.assertAlmostEqual(FlatnessSFX()(list(numpy.linspace(0.0, 1.0, 100))), 4.7500004768371582, 1e-6)


suite = allTests(TestFlatnessSfx)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
