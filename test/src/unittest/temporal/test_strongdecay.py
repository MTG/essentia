#!/usr/bin/env python

from essentia_test import *
from math import sqrt
from math import fabs

class TestStrongDecay(TestCase):

    def testEmpty(self):
        self.assertComputeFails(StrongDecay(), [])

    def testOne(self):
        self.assertComputeFails(StrongDecay(), [1])

    def testFlat(self):
        signal = [.5]*10
        centroid = sum(range(len(signal))) / float(len(signal))
        relativeCentroid = centroid * (1 / 44100.)
        energy = sum([x**2 for x in signal])
        strongDecay = sqrt(energy/relativeCentroid)
        self.assertAlmostEqual(
                StrongDecay()(signal),
                strongDecay)

    def testZero(self):
        self.assertComputeFails(StrongDecay(), [0]*10)

    def testSanity(self):
        growingSignal = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
        decayingSignal = [1,.9,.8,.7,.6,.5,.4,.3,.2,.1]

        self.assertTrue(StrongDecay()(growingSignal) <
                        StrongDecay()(decayingSignal))

    def testRegression(self):
        self.assertAlmostEqual(StrongDecay()([1,.9,.8,.7,.6,.5,.4,.3,.2,.1]), 237.897033691)

    def testZeroSumSignal(self):
        signal = [1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
        centroid = sum([i*fabs(x) for i,x in enumerate(signal)])/sum([fabs(x) for x in signal])
        energy = sum([x*x for x in signal])
        self.assertAlmostEqual(StrongDecay(sampleRate=1)(signal),
                               sqrt(float(energy)/float(centroid)))

suite = allTests(TestStrongDecay)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
