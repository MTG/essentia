#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/



from essentia_test import *
from numpy import sin, pi, mean, var
from numpy.random import uniform

class TestPitchSalience(TestCase):

    def testWhiteNoise(self):
        from numpy.random import uniform
        sig = array(uniform(size=10000))
        spec = Spectrum()(sig)
        self.assertTrue(PitchSalience()(spec) < .2)

    def testRealSignal(self):
        sr = 8000
        # Pure tone:
        sig = [sin(2*pi*440.0*i/float(sr)) for i in range(sr)]
        sineSalience = mean(self.computePitchSalience(sig,sr))

        # Sine weep:
        sig = [sin(2*pi*(20+440.0*i/float(sr))*i/float(sr)) for i in range(sr)]
        sweepSalience = mean(self.computePitchSalience(sig,sr))

        # white noise salience:
        sig = array(uniform(size=sr))
        noiseSalience = mean(self.computePitchSalience(sig,sr))

        # square wave salience
        stepSize = 40;
        step = 0.5;
        sig = []
        for i in range(0, sr, stepSize):
            step*=-1;
            sig += [step]*stepSize
        sqrSalience = mean(self.computePitchSalience(sig,sr))

        # varying sqr wave:
        stepSize = 10;
        step = 0.5;
        sig = []
        for i in range(0, sr, int(stepSize)):
            step*=-1;
            stepSize = 10 + 100*i/sr
            sig += [step]*stepSize
        varSqrSalience = mean(self.computePitchSalience(sig,sr))

        #print "noise: ", noiseSalience
        #print "sine:", sineSalience
        #print "sweep:", sweepSalience
        #print "square:", sqrSalience
        #print "var square:", varSqrSalience
        self.assertTrue(sineSalience  < sweepSalience)
        self.assertTrue(sweepSalience < noiseSalience)
        self.assertTrue(noiseSalience < varSqrSalience)
        self.assertTrue(varSqrSalience < sqrSalience)



    def computePitchSalience(self, signal, sampleRate):
        frames = FrameGenerator(signal)
        w = Windowing()
        spectrum = Spectrum()
        ps = PitchSalience(sampleRate=sampleRate,
                           lowBoundary=100,
                           highBoundary=sampleRate/2-1)
        salience = []
        for frame in frames:
            salience.append(ps(spectrum(w(frame))))
        return salience


    def testHarmonicSound(self):
        spectrum = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        sampleRate = len(spectrum)*2

        self.assertAlmostEqual(
                PitchSalience(sampleRate = sampleRate,
                              lowBoundary = .1*sampleRate/2,
                              highBoundary = .9*sampleRate/2)(spectrum),
                0.8888888,
                1e-6)

    def testInvalidParameter(self):
        # low > high boundary
        self.assertConfigureFails(
                PitchSalience(),
                {'lowBoundary': 100, 'highBoundary': 99})

    def testFullSpectrum(self):
        # arbitrary spectrum
        spectrum = [4,5,7,2,1,4,5,6,10]
        sampleRate = len(spectrum)*2

        self.assertAlmostEqual(
                PitchSalience(sampleRate=sampleRate,
                              lowBoundary=1,
                              highBoundary=sampleRate/2-1)(spectrum), 0.68014699220657349)

    def testEqualBoundariesOnPeak(self):
        # arbitrary spectrum
        spectrum = [4,0,0,0,0,0,0,0,0,0]*2205

        self.assertTrue(PitchSalience(lowBoundary=40, highBoundary=40)(spectrum) > .9)

    def testEqualBoundariesOffPeak(self):
        # arbitrary spectrum
        spectrum = [4,0,0,0,0,0,0,0,0,0]*2205

        self.assertTrue(PitchSalience(lowBoundary=41, highBoundary=41)(spectrum) < .1)

    def testSilence(self):
        self.assertEquals(PitchSalience()([0]*1024), 0)

    def testEmpty(self):
        self.assertComputeFails(PitchSalience(), [])

    def testBigHighBoundary(self):
        self.assertConfigureFails(PitchSalience(), {'highBoundary':22050})
        self.assertConfigureFails(PitchSalience(), {'highBoundary':22051})

    def testBigLowAndHighBoundary(self):
        self.assertConfigureFails(PitchSalience(), {'lowBoundary':22050,
                                                    'highBoundary':22051})


suite = allTests(TestPitchSalience)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
