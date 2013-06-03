#!/usr/bin/env python

from essentia_test import *


class TestRollOff(TestCase):

    def testEmpty(self):
        self.assertComputeFails(RollOff(), [])

    def testZeros(self):
        self.assertAlmostEqual(RollOff()([0]*101), 0)

    def testOne(self):
        self.assertComputeFails(RollOff(), [100])

    def testWhiteNoise(self):
        result = RollOff( cutoff=.85 )( [1]*101 )
        self.assertAlmostEqual(result, 44100/2 * .85)

    def testPinkNoise(self):
        input = [1./x for x in xrange(1,44101)]
        result = RollOff( cutoff=.85 )( input )
        self.assertAlmostEqual(result, 1.50003397465)

    def testMinCutoff(self):
        result = RollOff( cutoff=.000001 )( [1]*101 )
        self.assertAlmostEqual(result, 0)

    def testMaxCutoff(self):
        result = RollOff( cutoff=.999999 )( [1]*101 )
        self.assertAlmostEqual(result, 44100/2)

    def testNormalization(self):
        result = RollOff( sampleRate=200, cutoff=.5 )( [1]*101 )
        self.assertAlmostEqual(result, 50)

    def testRegression(self):
        input = readVector( join(filedir(), 'rolloff', 'input.txt') )
        expected = readVector( join(filedir(), 'rolloff', 'output.txt') )[0]

        result = RollOff(cutoff=.85)(input)
        self.assertAlmostEqual(result, expected)


suite = allTests(TestRollOff)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
