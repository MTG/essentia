#!/usr/bin/env python

from essentia_test import *
from math import *

class TestLowPass(TestCase):

    def testRegression(self):
        sr = 44100.
        pi2 = 2*pi
        signal = [.25*cos(t*pi2*5/sr) + \
                  .25*cos(t*pi2*50/sr) + \
                  .25*cos(t*pi2*500./sr) + \
                  .25*cos(t*pi2*5000./sr)
                  for t in range(44100)]

        filteredSignal = LowPass(cutoffFrequency=1000)(signal)

        s = Spectrum()(signal)
        sf = Spectrum()(filteredSignal)

        for i in range(1000):
            if s[i] > 10:
                self.assertTrue(sf[i] > 0,5*s[i])

        for i in range(1001, len(sf)):
            if s[i] > 10:
                self.assertTrue(sf[i] < 0.5*s[i])


suite = allTests(TestLowPass)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
