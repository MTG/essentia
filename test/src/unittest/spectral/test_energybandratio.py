#!/usr/bin/env python

from essentia_test import *


class TestEnergyBandRatio(TestCase):

    def testRegression(self):
        spectrum = readVector(filename = join(filedir(), 'energybandratio', 'input.txt'))
        expected = readValue(filename = join(filedir(), 'energybandratio', 'output.txt'))

        ebr = EnergyBandRatio(startFrequency = 0.0,
                              stopFrequency = 100.0,
                              sampleRate = 44100)

        self.assertAlmostEqual(ebr(spectrum), expected)
            

    def testZero(self):
        ebr = EnergyBandRatio(startFrequency = 0.0,
                              stopFrequency = 100.0,
                              sampleRate = 1)
        self.assertEqual(ebr(zeros(1000)), 0)


    def testInvalidParam(self):
        ebr = EnergyBandRatio()
        self.assertConfigureFails(ebr, { 'startFrequency': -25 })
        self.assertConfigureFails(ebr, { 'startFrequency': 100,
                                         'stopFrequency': 80 })

    def testInvalidInput(self):
        self.assertComputeFails(EnergyBandRatio(), [])


suite = allTests(TestEnergyBandRatio)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
