#!/usr/bin/env python

from essentia_test import *
from math import log10

class TestLeq(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(Leq(), input)

    def testSilence(self):
        input = [0]*100
        self.assertEqual(Leq()(input), -90)

    def testOne(self):
        input = [0]
        self.assertEqual(Leq()(input), -90)

        input = [100]
        self.assertAlmostEqual(Leq()(input), lin2db(10000.0))

    def testRegression(self):
        input = [45, 78, 1, 5, .1125, 1.236, 10.25, 100, 9, 78]
        expected = 10.0*log10(sum([x**2 for x in input]) / len(input))
        self.assertAlmostEqual(Leq()(input), expected)

    def testFullScaleSquare(self):
        sr = 44100
        freq = 1000
        step = 0.5*sr/freq
        size = 1*sr
        val = 1
        square = zeros(size)
        for i in range(size):
            square[i] = val
            if i%step < 1.0 :
                val *= -1
        self.assertAlmostEqual(Leq()(square), 0.0)


suite = allTests(TestLeq)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
