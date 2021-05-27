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

    def testZeros(self):
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic()(zeros(1024)))   

    def testOnes(self):
        referenceTonic =108.86099243164062     
        tonic   = TonicIndianArtMusic()(ones(1024))
        self.assertAlmostEqual(referenceTonic, tonic, 8)

    def testNegativeInput(self):
        tonic = TonicIndianArtMusic()([-1]*1024)
        referenceTonic =108.86099243164062     
        tonic   = TonicIndianArtMusic()([-1]*1024)
        self.assertAlmostEqual(referenceTonic, tonic, 8)

    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                            sampleRate = 44100)()
        referenceTonic = 102.74                                       
        tonic = TonicIndianArtMusic()(audio)
        self.assertAlmostEqual(referenceTonic, tonic, 6)

    def testMinMaxMismatch(self):
        frameSize = 2048
        signalSize = 15 * frameSize
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 99.1* 2*math.pi)
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic(minTonicFrequency=190, maxTonicFrequency=11)(x))

    def testBelowMinimumTonic(self):
        # generate test signal 99 Hz, and put minFreq as 100 Hz in the TonicIndianArtMusic
        defaultSampleRate = 44100
        frameSize = 2048
        signalSize = 15 * frameSize
        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 99* 2*math.pi)
        self.assertRaises(EssentiaException, lambda: TonicIndianArtMusic(minTonicFrequency=100)(x))   

    def testRegressionSyntheticSignal(self):

        # generate test signal concatenating major scale notes.
        defaultSampleRate = 44100
        frameSize = 2048
        signalSize = 15 * frameSize
        # Concat 3 sine waves together of different frequencies

        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 124 * 2*math.pi)
        y = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 100 * 2*math.pi)
        z = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 80 * 2*math.pi)
        scale = concatenate([x, y, z])

        tiam = TonicIndianArtMusic(minTonicFrequency=50, maxTonicFrequency=111)
        tonic  = tiam(scale)        
        # Check that tonic is above minTonicFrequency
        self.assertGreater(tonic, 50)
        # Check that tonic is below highest frequency in signal
        self.assertGreater(124, tonic)

        ### Make a (unharmonic) chord
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 124 * 2*math.pi)
        y = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 100 * 2*math.pi)
        z = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 80 * 2*math.pi)
        # This signal is a "major scale ladder"
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
