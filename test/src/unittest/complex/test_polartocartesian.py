#!/usr/bin/env python

# Copyright (C) 2006-2009 Music Technology Group (MTG)
#                         Universitat Pompeu Fabra


from essentia_test import *
from math import sin, cos

class TestPolarToCartesian(TestCase):

    def testEmpty(self):
        mags = []
        phases = []

        self.assertEqualVector(PolarToCartesian()(mags, phases), [])

    def testRegression(self):
        mags = [1, 4, 1.345, 0.321, -4]
        phases = [.45, 3.14, 2.543, 6.42, 1]

        expected = []
        for i in range(len(mags)):
            expected.append(mags[i] * cos(phases[i]) + (mags[i] * sin(phases[i]))*1j)

        self.assertAlmostEqualVector(PolarToCartesian()(mags, phases), expected, 1e-6)

    def testDiffSize(self):
        mags = [1]
        phases = [3, 4]
        self.assertComputeFails(PolarToCartesian(), mags, phases)


suite = allTests(TestPolarToCartesian)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
