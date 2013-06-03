#!/usr/bin/env python

from essentia_test import *

class TestLarm(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(Larm(),input)

    def testZero(self):
        input = zeros(44100)
        self.assertEquals(Larm()(input), -100)

    def testZerodB(self):
        input = ones(44100)
        self.assertEquals(Larm(attackTime=0,releaseTime=0)(input), 0.0)

    def testOneElement(self):
        input = [1]
        self.assertValidNumber(Larm()(input))

    def testNegativePower(self):
        input = ones(100)
        self.assertValidNumber(Larm(power=-1)(input))


    def testRegression(self):
        from numpy import sin, pi
        freq = 440.0
        attTime = 0.05 # attack time in s
        relTime = 0.1  # realease time in s
        silTime = 0.2 # silence time in s
        sr = 44100
        duration = 1 # time in s
        # first put some silence:
        input = [0]*(int)(silTime*sr)

        # attack:
        for i in range(int(attTime*sr)):
            input.append(float(i)/(attTime*sr)*sin(2.0*pi*freq*i/sr))

        # sustain:
        for i in range(int(attTime*sr), int((duration-relTime)*sr)):
            input.append(sin(2.0*pi*freq*i/sr))

        # release:
        for i in range(int(relTime*sr)):
            input.append((1-(float(i)/(relTime*sr)))*sin(2.0*pi*freq*i/sr))

        self.assertAlmostEqual(Larm()(input),-1.6623098850, 1e-3)

    def testInvalidParam(self):
        self.assertConfigureFails(Larm(),{'sampleRate':0})
        self.assertConfigureFails(Larm(),{'attackTime':-1})
        self.assertConfigureFails(Larm(),{'releaseTime':-1})


suite = allTests(TestLarm)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
