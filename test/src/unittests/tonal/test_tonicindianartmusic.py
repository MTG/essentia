#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Afextentro General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Afextentro GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/



from essentia_test import *
from numpy import sin, float32, pi, arange, mean, log2, floor, ceil

class TestTonicIndianArtMusic(TestCase):

    def testEmpty(self):
        tonic   = TonicIndianArtMusic()([])
        self.assertEqual(tonic, 0)

    def testZero(self):
        tonic   = TonicIndianArtMusic()([0])
        self.assertAlmostEqual(tonic, 108, 2)

    def testOnes(self):
        tonic   = TonicIndianArtMusic()([1]*1024)
        print(tonic)

    def testNegativeInput(self):
        # Negative vlues should be set to 0
        tonic = TonicIndianArtMusic()([-1]*1024)
        print(tonic)

    def testInvalidParam(self):
        self.assertConfigureFails(TonicIndianArtMusic(), { 'binResolution': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'frameSize': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'harmonicWeight': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'hopSize': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'magnitudeCompression': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'magnitudeThreshold': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'maxTonicFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'minTonicFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberHarmonics': 0 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberSaliencePeaks': 0})
        self.assertConfigureFails(TonicIndianArtMusic(), { 'referenceFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'sampleRate':   -1 })

    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                                           sampleRate = 44100)()
        tonicEst = TonicIndianArtMusic()(audio)
        print(tonicEst)

    def testSyntheticTonicIndianArtMusic(self):
        fs = 100  #Hz
        f0 = 100  #Hz
        extent = 5 #Hz
        vibrato = 5 

        x = [f0] * 1024 +  extent * sin(2 * pi * vibrato * arange(1024) / fs)

        tonicEst = TonicIndianArtMusic(sampleRate=fs)(x.astype(float32))
        print(tonicEst)

    def testFrequencyDetectionThreshold(self):
        #  TonicIndianArtMusic Min and Max default values are 3 and 8Hz so
        #  this values shouldn't be detected.

        fs = 100  #Hz
        f0 = 100  #Hz
        extent  = 5  #Hz
        vibrato = 3 #Hz
        x = [f0] * 1024 +   extent* sin(2 * pi * vibrato  * arange(1024) / fs)
        print(x)
        print("Segmentation fault in Freq. Detectio Threshold")
        #tonicEst = TonicIndianArtMusic(sampleRate=fs)(x.astype(float32))
        #self.assertEqual(0, tonicEst.all())

        extent = 9 #Hz
        x = [f0] * 1024 +  extent * sin(2 * pi * tonic * arange(1024) / fs)
        print(x)
        #tonicEst = TonicIndianArtMusic(sampleRate=fs)(x.astype(float32))
        #self.assertEqual(0, tonicEst.all())


    def testExtendDetectionThreshold(self):
        #  Extend Min and Max default values are 50 and 250 cents so
        #  this values  shouldn't be detected.
        fs = 100  #Hz
        f0 = 100  #Hz
        x = ceil(f0 * (2**(250/1200.) -1) / (2**(250/1200.) + 1))
        extent = 5 #Hz
        #x2 = [f0] * 1024 +  extent * sin(2 * pi * tonic * arange(1024) / fs)
        #print(x2)
        #tonicEst = TonicIndianArtMusic(sampleRate=fs)(x.astype(float32))
        #print(tonicEst)
        #extent  = floor(f0 * (2**(50/1200.) -1) / (2**(50/1200.) + 1))
        #x = [f0] * 1024 + extent  * sin(2 * pi * tonic * arange(1024) / fs)
        #tonicEst = TonicIndianArtMusic(sampleRate=fs)(x.astype(float32))
        #print(tonicEst)

suite = allTests(TestTonicIndianArtMusic)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
