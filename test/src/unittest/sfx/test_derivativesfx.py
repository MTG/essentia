#!/usr/bin/env python

from essentia_test import *
import numpy

class TestDerivativeSfx(TestCase):

    def testEmpty(self):
        self.assertComputeFails(DerivativeSFX(), [])

    def testZero(self):
        self.assertEqual(DerivativeSFX()([0]*100), (0.0, 0.0))

    def testOne(self):
        self.assertEqual(DerivativeSFX()([1234]), (1.0, 1234))

    def testAscending(self):
        derAvAfterMax, maxDerBeforeMax = DerivativeSFX()(list(numpy.linspace(0.0, 1.0, 100)))
        self.assertEqual(derAvAfterMax, maxDerBeforeMax)

    def testDescending(self):
        self.assertEqual(DerivativeSFX()(list(numpy.linspace(1.0, 0.0, 100))), (0.0, 1.0))

    def testRegression(self):
        input = list(numpy.linspace(0.0, 1.0, 100)) + list(numpy.linspace(1.0, 0.0, 100))
        self.assertAlmostEqualVector(DerivativeSFX()(input), [-0.0194097850471735, 0.010101020336151123])

suite = allTests(TestDerivativeSfx)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

