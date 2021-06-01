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
from numpy import sin, float32, pi, arange, mean, log2, floor, ceil, math, concatenate
import numpy as np

class TestTonicIndianArtMusic(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(TonicIndianArtMusic(), { 'binResolution': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'frameSize': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'binResolution': 0 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'frameSize': 0 })        
        self.assertConfigureFails(TonicIndianArtMusic(), { 'harmonicWeight': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'harmonicWeight': 0 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'harmonicWeight': 1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'hopSize': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'magnitudeCompression': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'magnitudeThreshold': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'maxTonicFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'minTonicFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberHarmonics': 0 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberHarmonics': -1 })        
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberSaliencePeaks': 0})
        self.assertConfigureFails(TonicIndianArtMusic(), { 'referenceFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'sampleRate':   -1 })

    def testEmpty(self):
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic()([]))   

    def testSilence(self):
        silence = np.zeros(int(np.abs(np.random.randn()) * 30. * 44100))
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic()(silence))

    def testOnes(self):
        referenceTonic =108.86
        tonic   = TonicIndianArtMusic()(ones(4096))
        self.assertAlmostEqual(tonic,referenceTonic, 6)

    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                            sampleRate = 44100)()
        referenceTonic = 102.74                                       
        tonic = TonicIndianArtMusic()(audio)
        self.assertAlmostEqual( tonic, referenceTonic, 6)
        start_zero = np.zeros(int(np.abs(np.random.randn()) * 30. * 44100))
        end_zero = np.zeros(int(np.abs(np.random.randn()) * 30. * 44100))
        # Check result is the same with appended silences
        real_audio = np.hstack([start_zero, audio, end_zero])
        tonic = TonicIndianArtMusic()(real_audio)
        self.assertAlmostEqual( tonic, referenceTonic, 6)

    def testWhiteNoise(self):
        from numpy.random import uniform
        sig = array(uniform(size=10000))
        tonic   = TonicIndianArtMusic()(sig)
        # Sanity check to see if result is greater than or equal to referenceFrequency
        # Check that tonic is below maxTonicFrequency
        self.assertGreater(375, tonic)
        # Check that tonic is above minTonicFrequency        
        self.assertGreater(tonic,100)
        # Sanity check for lowest possible reference frequency in this case
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic(referenceFrequency=1)(sig))

    def testMinMaxMismatch(self):
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic(minTonicFrequency=100, maxTonicFrequency=11)(ones(4096)))

    def testBelowMinimumTonic(self):
        frameSize = 2048
        signalSize = 15 * frameSize
        # generate test signal 99 Hz, and put minTonicFreq as 100 Hz in the TonicIndianArtMusic
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 99* 2*math.pi)
        self.assertRaises(EssentiaException, lambda: TonicIndianArtMusic(minTonicFrequency=100, maxTonicFrequency=375)(x))   

    def testAboveMaxTonic(self):
        frameSize = 2048
        signalSize = 15 * frameSize
        # generate test signal 101 Hz, and put maxTonicFreq as 100 Hz in the TonicIndianArtMusic        
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 101* 2*math.pi)        
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic(minTonicFrequency=99, maxTonicFrequency=100)(x))
 
    def testRegressionSyntheticSignal(self):
        # generate a test signal concatenating different frequencies
        frameSize = 2048
        signalSize = 15 * frameSize

        # Concat 3 sine waves together of different frequencies
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 124 * 2*math.pi)
        y = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 100 * 2*math.pi)
        z = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 80 * 2*math.pi)
        mix = concatenate([x, y, z])

        # tiam = acronym for "Tonic Indian Art Music"
        tiam = TonicIndianArtMusic(minTonicFrequency=50, maxTonicFrequency=111)
        tonic  = tiam(mix)        
        # Check that tonic is above minTonicFrequency
        self.assertGreater(tonic, 50)
        # Check that tonic is below highest frequency in signal
        self.assertGreater(124, tonic)

        ### Make a (unharmonic) chord
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 124 * 2*math.pi)
        y = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 100 * 2*math.pi)
        z = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 80 * 2*math.pi)
        chord =x+y+z

        tiam = TonicIndianArtMusic(minTonicFrequency=50, maxTonicFrequency=111)
        tonic  = tiam(chord)         
        
        # Check that tonic is above min frequency in signal
        self.assertGreater(tonic, 80)
        # Check that tonic is below highest frequency in signal        
        self.assertGreater(124, tonic)

suite = allTests(TestTonicIndianArtMusic)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
