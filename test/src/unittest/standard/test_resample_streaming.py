#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import *

class TestResample_Streaming(TestCase):

    def resample(self, input, factor):
        if input: sr = len(input)
        else: sr = 44100
        resample = Resample(inputSampleRate = sr,
                            outputSampleRate = int(factor*sr),
                            quality = 0)
        pool = Pool()
        gen = VectorInput(input)
        gen.data >> resample.signal
        resample.signal >> (pool, 'signal')
        run(gen)
        if not pool.descriptorNames() : return []
        return pool['signal']

    def assertResults(self, input, expected, factor, epsilon=1e-5):
        result = self.resample(input, factor)
        self.assertEqual(len(result), len(expected))
        self.assertAlmostEqual(sum(result[200:]), sum(expected[200:]), epsilon)

    def testEmpty(self):
        self.assertEqualVector(self.resample([], 2), [])

    def testSingle(self):
        self.assertAlmostEqualVector(self.resample([1], 2), [1], 5e-2)

    def testDouble(self):
        sr = 44100
        factor = 2
        input = [1]*sr
        expected = [1]*(sr*factor-1)
        self.assertResults(input, expected, factor)

    def testOneAndHalf(self):
        sr = 44100
        factor = 1.5
        input = [1]*sr
        expected = [1]*int(sr*factor-1)
        self.assertResults(input, expected, factor)

    def testOne(self):
        sr = 44100
        factor = 1
        input = [1]*sr
        expected = [1]*int(sr*factor)
        self.assertResults(input, expected, factor)

    def testHalf(self):
        sr = 44100
        factor = .5
        input = [1]*sr
        expected = [1]*int(sr*factor-1)
        self.assertResults(input, expected, factor)

    def testThreeQuarters(self):
        sr = 44100
        factor = .75
        input = [1]*sr
        expected = [1]*int(sr*factor-1)
        self.assertResults(input, expected, factor)

    def testOneQuarter(self):
        sr = 44100
        factor = .25
        input = [1]*sr
        expected = [1]*int(sr*factor-1)
        self.assertResults(input, expected, factor)

    #def testLeftLimits(self):
    #    # SRC resampling capabilites are limited to the range [1/256, 256]
    #    sr = 44100
    #    factor = 1./254
    #    input = [1]*sr
    #    expected = [1]*int(sr*factor-1)
    #    self.assertResults(input, expected, factor)

    #def testRightLimits(self):
    #    # SRC resampling capabilites are limited to the range [1/256, 256]
    #    sr = 44100
    #    factor = 256
    #    input = [1]*10
    #    expected = [1]*int(10*factor-1)
    #    self.assertResults(input, expected, factor, 1e-1)

    #def testBeyondLeftLimits(self):
    #    # SRC resampling capabilites are limited to the range [1/256, 256]
    #    sr = 44100
    #    factor = 1./258
    #    input = [1]*258
    #    resample = Resample(inputSampleRate = sr,
    #                        outputSampleRate = int(factor*sr),
    #                        quality = 0)
    #    pool = Pool()
    #    gen = VectorInput(input)
    #    gen.data >> resample.signal
    #    resample.signal >> (pool, 'signal')
    #    self.assertRaises(RuntimeError, lambda: run(gen))

    #def testBeyondRightLimits(self):
    #    # SRC resampling capabilites are limited to the range [1/256, 256]
    #    sr = 44100
    #    factor = 258
    #    input = [1]*10
    #    resample = Resample(inputSampleRate = sr,
    #                        outputSampleRate = int(factor*sr),
    #                        quality = 0)
    #    pool = Pool()
    #    gen = VectorInput(input)
    #    gen.data >> resample.signal
    #    resample.signal >> (pool, 'signal')
    #    self.assertRaises(RuntimeError, lambda: run(gen))

suite = allTests(TestResample_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
