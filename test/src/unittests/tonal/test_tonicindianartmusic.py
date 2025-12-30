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
from numpy import sin, float32, pi, arange, mean, log2, floor, ceil, concatenate
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
        self.assertConfigureFails(TonicIndianArtMusic(), { 'magnitudeCompression': 2 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'magnitudeThreshold': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'maxTonicFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'minTonicFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberHarmonics': 0 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberHarmonics': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberSaliencePeaks': 0})
        self.assertConfigureFails(TonicIndianArtMusic(), { 'numberSaliencePeaks': 16})
        self.assertConfigureFails(TonicIndianArtMusic(), { 'referenceFrequency': -1 })
        self.assertConfigureFails(TonicIndianArtMusic(), { 'sampleRate':   -1 })

    def testEmpty(self):
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic()([]))

    def testSilence(self):
        # test 1 second of silence
        silence = np.zeros(44100)
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic()(silence))

    def testOnes(self):
        # Not a realistic test but useful for sanity checks/ regression checks.
        referenceTonic = 108.86
        tonic = TonicIndianArtMusic()(ones(1024))
        self.assertAlmostEqualFixedPrecision(tonic, referenceTonic, 2)

    # Full reference set of values can be sourced from dataset
    # Download https://compmusic.upf.edu/carnatic-varnam-dataset
    # See  file "tonics.yaml"
    #
    # vignesh: 138.59
    # This tonic corresponds to the following mp3 file.
    # "23582__gopalkoduri__carnatic-varnam-by-vignesh-in-abhogi-raaga.mp3'
    #
    #  copy this file into essentia/test/audio/recorded.


    def testRegressionVignesh(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/223582__gopalkoduri__carnatic-varnam-by-vignesh-in-abhogi-raaga.mp3'),
                            sampleRate = 44100)()

        # Reference tonic from YAML file is 138.59.  The measured is "138.8064422607422"
        referenceTonic = 138.59
        tonic = TonicIndianArtMusic()(audio)
        self.assertAlmostEqualFixedPrecision(tonic, referenceTonic, 0)

    def testRegression(self):
        # Regression test using existing vignesh audio file in "essentia/test/audio/recorded"
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                            sampleRate = 44100)()
        referenceTonic = 102.74
        tonic = TonicIndianArtMusic()(audio)
        self.assertAlmostEqualFixedPrecision( tonic, referenceTonic, 2)
        start_zero = np.zeros(int(44100))
        end_zero = np.zeros(int(44100))
        # Check result is the same with appended silences of constant length
        real_audio = np.hstack([start_zero, audio, end_zero])
        tonic = TonicIndianArtMusic()(real_audio)
        self.assertAlmostEqualFixedPrecision(tonic, referenceTonic, 2)

    def testMinMaxMismatch(self):
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic(minTonicFrequency=100,maxTonicFrequency=11)(ones(4096)))

    def testBelowMinimumTonic(self):
        signalSize = 15 * 2048
        # generate test signal 99 Hz, and put minTonicFreq as 100 Hz in the TonicIndianArtMusic
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 99* 2*pi)
        self.assertRaises(EssentiaException, lambda: TonicIndianArtMusic(minTonicFrequency=100,maxTonicFrequency=375)(x))

    def testAboveMaxTonic(self):
        signalSize = 15 * 2048
        # generate test signal 101 Hz, and put maxTonicFreq as 100 Hz in the TonicIndianArtMusic
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 101* 2*pi)
        self.assertRaises(RuntimeError, lambda: TonicIndianArtMusic(minTonicFrequency=99,maxTonicFrequency=100)(x))

    def testRegressionSyntheticSignal(self):
        # generate a test signal concatenating different frequencies
        signalSize = 15 * 2048

        # Concat 3 sine waves together of different frequencies
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 124 * 2*pi)
        y = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 100 * 2*pi)
        z = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 80 * 2*pi)
        mix = concatenate([x, y, z])

        # tiam = acronym for "Tonic Indian Art Music"
        tiam = TonicIndianArtMusic(minTonicFrequency=50, maxTonicFrequency=111)
        tonic  = tiam(mix)
        # Check that tonic is above minTonicFrequency
        self.assertGreater(tonic, 50)
        # Check that tonic is below highest frequency in signal
        self.assertGreater(124, tonic)

        ### Make a (unharmonic) chord
        x = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 124 * 2*pi)
        y = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 100 * 2*pi)
        z = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 80 * 2*pi)
        chord = x+y+z

        tiam = TonicIndianArtMusic(minTonicFrequency=50, maxTonicFrequency=111)
        tonic  = tiam(chord)

        # Check that tonic is above min frequency in signal
        self.assertGreater(tonic, 80)
        # Check that tonic is below highest frequency in signal
        self.assertGreater(124, tonic)

suite = allTests(TestTonicIndianArtMusic)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
