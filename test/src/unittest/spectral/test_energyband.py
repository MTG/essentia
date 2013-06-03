#!/usr/bin/env python

from essentia_test import *


class TestEnergyBand(TestCase):

    def testEmpty(self):
        self.assertComputeFails(EnergyBand(), [])

    def testZero(self):
        self.assertEqual(EnergyBand()(zeros(512)), 0)

    def testDefaultBand(self):
        stopCutoffFreq = 100.0
        size = 512
        sr = 44100
        stopBin = int(stopCutoffFreq*size/(0.5*sr) + 0.5)
        result  = sum(ones(stopBin+1))
        self.assertEqual(EnergyBand()(ones(size)), result)

    def testBand(self):
        startCutoffFreq = 115.0
        stopCutoffFreq = 880.0
        size = 512
        sr = 44100
        startBin = int(startCutoffFreq*size/(0.5*sr) + 0.5)
        stopBin  = int(stopCutoffFreq*size/(0.5*sr) + 0.5)
        result   = sum(ones(stopBin-startBin+1))
        self.assertEqual(EnergyBand(startCutoffFrequency=startCutoffFreq,
                                    stopCutoffFrequency=stopCutoffFreq)(ones(size)), result)

    def testInvalidParam(self):
        self.assertConfigureFails(EnergyBand(), {'startCutoffFrequency':800, 'stopCutoffFrequency':100})
        self.assertConfigureFails(EnergyBand(), {'startCutoffFrequency':100, 'stopCutoffFrequency':100})
        self.assertConfigureFails(EnergyBand(), {'sampleRate':0})
        self.assertConfigureFails(EnergyBand(), {'startCutoffFrequency':22050, 'stopCutoffFrequency':30000})
        self.assertConfigureFails(EnergyBand(), {'stopCutoffFrequency':22051})

    def testSameBand(self):
        size = 512
        sr = 44100
        result = 1
        self.assertEqual(EnergyBand(startCutoffFrequency=100,
                                    stopCutoffFrequency=102)(ones(size)), result)

suite = allTests(TestEnergyBand)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
