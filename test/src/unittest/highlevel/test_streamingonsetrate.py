#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import OnsetRate

class TestStreamingOnsetRate(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        sor = OnsetRate()
        p = Pool()

        gen.data >> sor.signal
        sor.onsetTimes >> (p, 'data')
        sor.onsetRate >> (p, 'onset.rate')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testZero(self):
        gen = VectorInput( [0]*10*1024 )
        sor = OnsetRate()
        p = Pool()

        gen.data >> sor.signal
        sor.onsetTimes >> (p, 'onset.times')
        sor.onsetRate >> (p, 'onset.rate')

        run(gen)

        self.assertEqual(p['onset.rate'], 0)


    def ImpulseTrain(self, frameSize, factor, precision):
        nFrames = 16
        inputData = [0]*nFrames*frameSize
        pos = factor*frameSize

        # after the first frame,every factor frames there will be an impulse of
        # type 1, 0.8, 0.6, 0.4, 0.2
        for i in xrange(len(inputData)):
            mod = i%pos
            if i > frameSize and mod < 5:
                inputData[i] = 0.5*(5-mod)/2.5 # impulse: 1, 0.8, 0.6, 0.4, 0.2

        size = int(nFrames/factor);
        expected = [0]*size
        for i in xrange(size):
          expected[i] = factor*(i+1)*frameSize/44100

        gen = VectorInput(inputData)
        onsetRate = OnsetRate()
        p = Pool()

        gen.data >> onsetRate.signal
        onsetRate.onsetTimes >> (p, 'onset.times')
        onsetRate.onsetRate >> (p, 'onset.rate')

        run(gen)

        self.assertAlmostEqual(p['onset.rate'], len(expected)/float(len(inputData))/44100.0, 0.05)

        self.assertAlmostEqualVector(p['onset.times'], expected, precision)


    def testImpulseTrain(self):
        for i in range(4):
            frameSize = 2**i * 1024

            for j in range(20):
                factor = .05*j + 5

                self.ImpulseTrain(frameSize, factor, 0.1)


suite = allTests(TestStreamingOnsetRate)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
