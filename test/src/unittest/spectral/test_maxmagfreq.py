#!/usr/bin/env python

from essentia_test import *

class TestMaxMagFreq(TestCase):

    def testEmpty(self):
        self.assertComputeFails(MaxMagFreq(), [])

    def testOne(self):
        self.assertComputeFails(MaxMagFreq(), [1])

    def testZeroFreq(self):
        self.assertAlmostEqual(
                MaxMagFreq()([10, 1, 2, 3]),
                0)

    def testRegression(self):
        self.assertAlmostEqual(
                MaxMagFreq()([3.55,-4.11,0.443,3.9555,2]),
                3 * 22050 / 4.)

    def testNonDefaultSampleRate(self):
        self.assertAlmostEqual(
                MaxMagFreq(sampleRate=10)([1,2,3,4,5]),
                4 * 5 / 4.)

    def testInvalidParam(self):
        self.assertConfigureFails(MaxMagFreq(), {'sampleRate' : 0})

suite = allTests(TestMaxMagFreq)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
