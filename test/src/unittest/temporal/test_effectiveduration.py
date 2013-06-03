#!/usr/bin/env python

from essentia_test import *
from random import randint


class TestEffectiveDuration(TestCase):

    def testEmpty(self):
        input = []
        self.assertEqual(EffectiveDuration()(input), 0.0)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(EffectiveDuration()(input), 0.)

    def testOne(self):
        input = [0.3]
        self.assertAlmostEqual(EffectiveDuration()(input), 1/44100.0)
        input = [0]
        self.assertAlmostEqual(EffectiveDuration()(input), 0)


        input = [100]
        self.assertAlmostEqual(EffectiveDuration()(input), 1/44100.0)

    def test30Sec(self):
        input = [randint(41, 100) for x in xrange(44100*30)]
        self.assertAlmostEqual(EffectiveDuration()(input), 30)

    def test15SecOf30Sec(self):
        input1 = [randint(41, 100) for x in xrange(44100*15)]
        input1[0] = 100 # to ensure that at least one element is 100
        input2 = [randint(0, 39) for x in xrange(44100*15)]
        input = input1 + input2

        self.assertAlmostEqual(EffectiveDuration()(input), 15)

    def testNegative20SecOf40Sec(self):
        # Note: this test assumes the thresholdRatio is 40%
        input1 = [randint(-100, -41) for x in xrange(44100*10)]
        input2 = [randint(0, 39) for x in xrange(44100*10)]
        input3 = [randint(41, 100) for x in xrange(44100*10)]
        input3[0] = 100 # to ensure that at least one element is 100
        input4 = [randint(-39, 0) for x in xrange(44100*10)]

        input = input1 + input2 + input3 + input4

        self.assertAlmostEqual(EffectiveDuration()(input), 20)

    def testBadSampleRate(self):
        self.assertConfigureFails(EffectiveDuration(), { 'sampleRate' : 0 })


suite = allTests(TestEffectiveDuration)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
