#!/usr/bin/env python

from essentia_test import *

class TestLoudness(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(Loudness(), input)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(Loudness()(input), 0)

    def testOne(self):
        input = [0]
        self.assertAlmostEqual(Loudness()(input), 0)

        input = [100]
        self.assertAlmostEqual(Loudness()(input), 478.63009232263852, 1e-6)

    def testRegression(self):
        input = [45, 78, 1, -5, -.1125, 1.236, 10.25, 100, 9, -78]
        self.assertAlmostEqual(Loudness()(input), 870.22171051882947, 1e-6)


suite = allTests(TestLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
