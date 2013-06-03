#!/usr/bin/env python

from essentia_test import *

class TestHighResolutionFeatures(TestCase):

    def testEmpty(self):
        # empty array should throw an exception cause it is not multiple of 12:
        self.assertComputeFails(HighResolutionFeatures(), [])

    def testZero(self):
        # array of zeros should output zeros
        self.assertEqualVector(HighResolutionFeatures()(zeros(120)), [0, 0, 0])

    def testPerfectlyTempered(self):
        nCents = 10
        nSemitones = 12
        size = nCents*nSemitones
        hpcp = zeros(size)
        # peaks at semitone position:
        for i in range(0, size, nCents):
            hpcp[i] = 1.0
        self.assertEqualVector(HighResolutionFeatures()(hpcp), [0, 0, 0])

    def testMaxDeviation(self):
        nCents = 10
        nSemitones = 12
        size = nCents*nSemitones
        hpcp = zeros(size)
        # peaks at semitone position + 0.5:
        for i in range(0, size, nCents/2):
            hpcp[i] = 1.0
        self.assertEqualVector(HighResolutionFeatures()(hpcp), [0.25, 0.5, 0.5])

    def testStreamingMaxDeviation(self):
        from essentia.streaming import HighResolutionFeatures as\
        strHighResolutionFeatures

        nCents = 10
        nSemitones = 12
        size = nCents*nSemitones
        hpcp = [0]*size
        # peaks at semitone position + 0.5:
        for i in range(0, size, nCents/2):
            hpcp[i] = 1

        gen = VectorInput([hpcp])
        hrf = strHighResolutionFeatures()
        pool = Pool()
        gen.data >> hrf.hpcp
        hrf.equalTemperedDeviation >> (pool, "deviation")
        hrf.nonTemperedEnergyRatio >> (pool, "energyRatio")
        hrf.nonTemperedPeaksEnergyRatio >> (pool, "peaksRatio")
        run(gen)

        self.assertEqual(pool['deviation'], 0.25)
        self.assertEqual(pool['energyRatio'], 0.5)
        self.assertEqual(pool['peaksRatio'], 0.5)


suite = allTests(TestHighResolutionFeatures)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
