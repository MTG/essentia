#!/usr/bin/env python

from essentia_test import *
from math import log10

class TestLogAttackTime(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(LogAttackTime(), input)

    def testSilence(self):
        input = [0]*100
        self.assertAlmostEqual(LogAttackTime()(input), -5)

    def testOne(self):
        input = [0]
        self.assertEqual(LogAttackTime()(input), -5)

        input = [100]
        self.assertEqual(LogAttackTime()(input), -5)

    def testInvalidStartStop(self):
        self.assertConfigureFails(
                LogAttackTime(),
                {'startAttackThreshold': .8, 'stopAttackThreshold': .2})

    def testImpulse(self):
        input = [1,1,1,1,1,10,1,1,1,1,1]

        self.assertEqual( LogAttackTime()(input), -5 )

    def testRegression(self):
        #       start                              stop
        #         |---------------------------------|
        input = [45, 78, 1, 5, .1125, 1.236, 10.25, 100, 9, 78]
        expected = log10(7/44100.)

        self.assertAlmostEqual(
                LogAttackTime(startAttackThreshold=.1,
                              stopAttackThreshold=.9)(input),
                expected)


    def testZero(self):
        input = [0]*1024
        self.assertEqual(LogAttackTime()(input), -5)


suite = allTests(TestLogAttackTime)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
