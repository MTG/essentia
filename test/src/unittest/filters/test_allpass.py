#!/usr/bin/env python

from essentia_test import *
from math import *

class TestAllPass(TestCase):

    def testRegressionOrder1(self):
        sr = 44100.
        pi2 = 2*pi
        signal = [.25*cos(t*pi2*5/sr) + \
                  .25*cos(t*pi2*50/sr) + \
                  .25*cos(t*pi2*500./sr) + \
                  .25*cos(t*pi2*5000./sr)
                  for t in range(44100)]

        filteredSignal = AllPass(order=1, cutoffFrequency=500, bandwidth=100)(signal)

        s = Spectrum()(signal)
        sf = Spectrum()(filteredSignal)

        for i in range(len(s)):
            if s[i] > 10:
                self.assertTrue(sf[i] > 0.99*s[i])

    def testRegressionOrder2(self):
        sr = 44100.
        pi2 = 2*pi
        signal = [.25*cos(t*pi2*5/sr) + \
                  .25*cos(t*pi2*50/sr) + \
                  .25*cos(t*pi2*500./sr) + \
                  .25*cos(t*pi2*5000./sr)
                  for t in range(44100)]

        filteredSignal = AllPass(order=2, cutoffFrequency=500, bandwidth=100)(signal)

        s = Spectrum()(signal)
        sf = Spectrum()(filteredSignal)

        for i in range(len(s)):
            if s[i] > 10:
                self.assertTrue(sf[i] > 0.99*s[i])


suite = allTests(TestAllPass)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
