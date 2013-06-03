#!/usr/bin/env python

from essentia_test import *

class TestPowerSpectrum(TestCase):

    def testRegression(self):
        input = readVector( join(filedir(), 'powerspectrum', 'input.txt') )
        expected = readVector( join(filedir(), 'powerspectrum', 'output.txt') )

        result = PowerSpectrum(size=len(input))(input)
        self.assertAlmostEqualVector(result, expected, 1e-3)

    def testEmpty(self):
        signal = []
        self.assertComputeFails(PowerSpectrum(), signal)

    def testOne(self):
        signal = [1]
        self.assertComputeFails(PowerSpectrum(), signal)

    def testTwo(self):
        signal = [-1.156,0]
        self.assertAlmostEqualVector(
            PowerSpectrum()(signal),
            [x**2 for x in Spectrum()(signal)],
            1e-6)

    def testSize(self):
        signal = range(100)
        self.assertAlmostEqualVector(
            PowerSpectrum(size=54)(signal),
            [x**2 for x in Spectrum()(signal)],
            1e-6)

    def testImpulse(self):
        size = 1024
        specSize = size/2+1
        signal = zeros(size);
        signal[0] = 1
        self.assertEquals(sum(PowerSpectrum()(signal)), specSize)

    def testZero(self):
        input = [0]*1024
        expected = [0]*513

        result = PowerSpectrum(size=len(input))(input)

        self.assertEqualVector(result, expected)

suite = allTests(TestPowerSpectrum)

if __name__ == '__main__':
    TextTestRunner(verboslaclacilacty=2).run(suite)
