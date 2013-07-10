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
from numpy import random, pi, sort

class TestDissonance(TestCase):

    def testEmpty(self):
        # silent frames should have no dissonance
        self.assertAlmostEqual(Dissonance()([],[]), 0)

    def testOne(self):
        pass

    def testDiffSizeInputs(self):
        self.assertComputeFails(Dissonance(), [1], [1,2])

    def testNotOrderedByFreq(self):
        self.assertComputeFails(Dissonance(), [1,3,2], [1,2,3])

    def testZeros(self):
        self.assertAlmostEqual(Dissonance()( range(44100), [0]*44100 ), 0)

    def testBadFreqs(self):
        self.assertAlmostEqual(Dissonance()( [0]*100, [1]*100 ), 0)

    def testNoRelevantFreqs(self):
        self.assertAlmostEqual(Dissonance()( [1,2,3, 10001, 10002], [.1]*5), 0)


    def testEnharmonic(self):
        size = 32
        fund_freq = 440
        harms = [fund_freq*(pi/3+ 3*float(random.rand(1))) for i in range(1, size+1)]
        harms[0] = fund_freq
        harms = array(sort(harms))
        mags = zeros(size)
        mags[0] = 1
        mags[1] = 0.833333
        mags[2] = 0.75
        mags[3] = 0.555555
        mags[7] = 0.211
        mags[15] = 0.05
        self.assertNotEquals(Dissonance()(harms, mags), 0)

    def testHarmonic(self):
        size = 32
        fund_freq = 440
        # harmonics:
        harms = [fund_freq*i for i in range(1, size+1)]

        # enharmonics:
        enharms = [fund_freq*(pi/3.0*(1+float(random.rand(1))/2.0)) for i in range(size)]
        enharms[0] = fund_freq
        enharms = array(sort(enharms))

        mags = zeros(size)
        mags[0] = 1
        mags[1] = 0.833333
        mags[2] = 0.75
        mags[3] = 0.555555
        mags[7] = 0.211
        mags[15] = 0.05

        self.assertTrue(Dissonance()(harms, mags) < Dissonance()(enharms, mags))

    def testTritone(self):
        # tritone == augmented fourth i.e. C-F#
        cFreq = 261.63
        cisFreq =523.25 #277.18
        fisFreq = 369.99
        semitoneFreqs = [cFreq, cisFreq] # C-C#
        tritoneFreqs = [cFreq, fisFreq] # C-F#
        spec = [1, 1]
        self.assertTrue(Dissonance()(tritoneFreqs, spec) > Dissonance()(semitoneFreqs, spec))

    def testSemitoneAndSemitoneOctaveHigher(self):
        # dissonance (roughness) only applies to tones close enough:
        cFreq = 261.63
        cisFreq = 277.18
        cisFreq1 = cisFreq*2 # 1 octave above
        semitoneFreqs = [cFreq, cisFreq] # C-C#
        semitoneFreqs1 = [cFreq, cisFreq1] # C-C1#
        spec = [1, 1]
        result1 = Dissonance()(semitoneFreqs, spec)
        result2 = Dissonance()(semitoneFreqs1, spec)
        self.assertEqual(result2, 0)
        self.assertTrue(result1 > result2)

    def testHigherOctave(self):
        # In general, roughness is more noticeable in lower octaves, thus
        cFreq = 261.63
        cisFreq = 277.18
        cFreq1 = cFreq*4     # 2 octave above
        cisFreq1 = cisFreq*4 # 2 octave above
        semitoneFreqs = [cFreq, cisFreq] # C-C#
        semitoneFreqs1 = [cFreq1, cisFreq1] # C2-C2#
        spec = [1, 1]
        self.assertTrue(Dissonance()(semitoneFreqs1, spec) <=  Dissonance()(semitoneFreqs, spec))

    def testPerfectConsonance(self):
        cFreq = 261.63
        tones = [cFreq, cFreq] # C-C1#
        spec = [1, 1]
        self.assertAlmostEqual(Dissonance()(tones, spec), 0)

    def testOctaveAndFifth(self):
        # octave and fifths should be consonant or else fifth should be more
        # dissonant
        cFreq = 110
        cFreq2 = cFreq*2
        cFreq5 = cFreq*3/2

        octave = [cFreq, cFreq2]
        fifth = [cFreq, cFreq5]
        spec = [1, 1]
        dis1 = Dissonance()(octave, spec)
        dis2 = Dissonance()(fifth,spec)
        self.assertTrue(Dissonance()(octave, spec) <= Dissonance()(fifth,spec))

    def testRegression(self):
        sampleRate = 44100
        filename = join(testdata.audio_dir, 'recorded', \
                       'musicbox.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()

        fc = FrameCutter(frameSize=4096, hopSize=512)
        windower = Windowing(type='blackmanharris62')
        specAlg = Spectrum(size=4096)
        sPeaksAlg = SpectralPeaks(sampleRate = sampleRate,
                                  maxFrequency = sampleRate/2,
                                  minFrequency = 0,
                                  orderBy = 'frequency')
        dissonanceAlg = Dissonance()

        # Calculate the average dissonance over all frames of audio
        frame = fc(audio)
        dissSum = 0
        count = 0
        while len(frame) != 0:
            spectrum = specAlg(windower(frame))
            peaks = sPeaksAlg(spectrum)
            dissSum += dissonanceAlg(*peaks)

            count += 1
            frame = fc(audio)

        dissAvg = float(dissSum) / float(count)
        self.assertAlmostEqual(dissAvg, 0.475740219278, 1e-6)

suite = allTests(TestDissonance)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
