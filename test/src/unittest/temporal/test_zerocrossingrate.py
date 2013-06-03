#!/usr/bin/env python

from essentia_test import *

class TestZeroCrossingRate(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(ZeroCrossingRate(), input)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testOne(self):
        input = [0]
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

        input = [100]
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testAllPositive(self):
        input = [1]*100
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testAllNegative(self):
        input = [-1]*100
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testRegression(self):
        input = [45, 78, 1, -5, -.1125, 1.236, 10.25, 100, 9, -78]
        self.assertAlmostEqual(ZeroCrossingRate()(input), 3./10.)

    def testRealCase(self):
        # a 5000 cycle sine wave should cross the zero line 10000 times
        filename=join(testdata.audio_dir, 'generated', 'sine_440_5000period.wav')
        signal = MonoLoader(filename=filename)()
        wavzcr = ZeroCrossingRate(threshold=0.0)(signal)*len(signal)

        filename=join(testdata.audio_dir, 'generated', 'sine_440_5000period.mp3')
        signal = MonoLoader(filename=filename)()
        mp3zcr = ZeroCrossingRate(threshold=0.01)(signal)*len(signal)

        self.assertAlmostEqual(wavzcr, 10000)
        self.assertAlmostEqual(wavzcr, mp3zcr)


suite = allTests(TestZeroCrossingRate)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
