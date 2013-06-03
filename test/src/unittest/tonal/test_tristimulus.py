#!/usr/bin/env python

from essentia_test import *

class TestTristimulus(TestCase):

    def testZeroMag(self):
        mags = [0,0,0,0,0]
        freqs = [23, 500, 3200, 9000, 10000]

        self.assertEqualVector(
            Tristimulus()(freqs, mags),
            [0,0,0])


    def test3Freqs(self):
        mags = [1,2,3]
        freqs = [100, 200, 300]

        self.assertAlmostEqualVector(
            Tristimulus()(freqs, mags),
            [0.1666666667, 0, 0])


    def test4Freqs(self):
        mags = [1,2,3,4]
        freqs = [100, 435, 6547, 24324]

        self.assertAlmostEqualVector(
            Tristimulus()(freqs, mags),
            [.1, .9, 0])

    def test5Freqs(self):
        mags = [1,2,3,4,5]
        freqs = [100, 324, 5678, 5899, 60000]

        self.assertAlmostEqualVector(
            Tristimulus()(freqs, mags),
            [0.0666666667, .6, 0.33333333333])

    def testFrequencyOrder(self):
        freqs = [1,2,1.1]
        mags = [0,0,0]
        self.assertComputeFails(Tristimulus(), freqs, mags)

    def testFreqMagDiffSize(self):
        freqs = [1]
        mags = []
        self.assertComputeFails(Tristimulus(), freqs, mags)

    def testEmpty(self):
        freqs = []
        mags = []
        self.assertEqualVector(Tristimulus()([],[]), [0,0,0])


suite = allTests(TestTristimulus)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
