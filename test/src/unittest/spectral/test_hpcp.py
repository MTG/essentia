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

class TestHPCP(TestCase):

    def testEmpty(self):
        hpcp = HPCP()([], [])
        self.assertEqualVector(hpcp, [0.]*12)

    def testZeros(self):
        hpcp = HPCP()([0]*10, [0]*10)
        self.assertEqualVector(hpcp, [0.]*12)

    def testSin440(self):
        # Tests whether a real audio signal of one pure tone gets read as a
        # single semitone activation, and gets read into the right pcp bin
        sampleRate = 44100
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_0db.wav'),
                           sampleRate = sampleRate)()
        speaks = SpectralPeaks(sampleRate = sampleRate,
                               maxPeaks = 1,
                               maxFrequency = sampleRate/2,
                               minFrequency = 0,
                               magnitudeThreshold = 0,
                               orderBy = 'magnitude')
        (freqs, mags) = speaks(Spectrum()(audio))
        hpcp = HPCP()(freqs, mags)
        self.assertEqualVector(hpcp, [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    def testAllSemitones(self):
        # Tests whether a spectral peak output of 12 consecutive semitones
        # yields a HPCP of all 1's
        tonic = 440
        freqs = [(tonic * 2**(x/12.)) for x in range(12)]
        mags = [1] * 12
        hpcp = HPCP()(freqs, mags)
        self.assertEqualVector(hpcp, [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])

    def testSubmediantPosition(self):
        # Make sure that the submediant of a key based on 440 is in the
        # correct location (submediant was randomly selected from all the
        # tones)
        tonic = 440
        submediant = tonic * 2**(9./12.)
        hpcp = HPCP()([submediant], [1])

        self.assertEqualVector(hpcp, [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.])

    def testMaxShifted(self):
        # Tests whether a HPCP reading with only the dominant semitone
        # activated is correctly shifted so that the dominant is at the
        # position 0
        tonic = 440
        dominant = tonic * 2**(7./12.)
        hpcp = HPCP(maxShifted=True)([dominant], [1])

        self.assertEqualVector(hpcp, [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    def chordHelper(self, half_steps, tunning, strength):
        notes = [tunning*(2.**(half_steps[i]/12.)) for i in range(len(half_steps))]
        hpcp = HPCP(maxShifted=False)([notes[0], notes[1], notes[2]], strength)
        for i in range(len(hpcp)):
            if i in half_steps: self.assertTrue(hpcp[i]>0)
            elif (i - 12) in half_steps: self.assertTrue(hpcp[i]>0)
            else: self.assertEqual(hpcp[i], 0)

    def testChord(self):
        tunning = 440
        AMajor = [0, 4, 7] # AMajor = A4-C#5-E5
        self.chordHelper(AMajor, tunning, [1,1,1])
        CMajor = [3, -4, -2] # CMajor = C5-F4-G4
        self.chordHelper(CMajor, tunning, [1,1,1])
        CMajor = [-4, 3, -2] # CMajor = C5-F4-G4
        self.chordHelper(CMajor, tunning, [1,0.5,0.2])
        CMajor = [-4, -2, 3] # CMajor = C5-F4-G4
        self.chordHelper(CMajor, tunning, [1,0.5,0.2])
        CMajor = [3, 8, 10] # CMajor = C5-F5-G5
        self.chordHelper(CMajor, tunning, [1,0.5,0.2])
        AMinor = [0, 3, 7] # AMinor = A4-C5-E5
        self.chordHelper(AMinor, tunning, [1,0.5,0.2])
        CMinor = [3, 6, 10] # CMinor = C5-E5-G5
        self.chordHelper(CMinor, tunning, [1,0.5,0.2])


    # Test of various parameter logical bounds

    def testLowFrequency(self):
        hpcp = HPCP(minFrequency=100, maxFrequency=1000)([99], [1])
        self.assertEqualVector(hpcp, [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    def testHighFrequency(self):
        hpcp = HPCP(minFrequency=100, maxFrequency=1000)([1001], [1])
        self.assertEqualVector(hpcp, [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    def testSmallMinRange(self):
        self.assertConfigureFails(HPCP(), {'minFrequency':1, 'splitFrequency':200})

    def testSmallMaxRange(self):
        self.assertConfigureFails(HPCP(), {'maxFrequency':1199, 'splitFrequency':1000})

    def testSmallMinMaxRange(self):
        self.assertConfigureFails(HPCP(), {'bandPreset':False, 'maxFrequency':200, 'minFrequency':1})

    def testSizeNonmultiple12(self):
        self.assertConfigureFails(HPCP(), {'size':13})

    def testHarmonics(self):
        # Regression test for the 'harmonics' parameter
        tone = 100. # arbitrary frequency [Hz]
        freqs = [tone, tone*2, tone*3, tone*4]
        mags = [1]*4

        hpcpAlg = HPCP(minFrequency=50, maxFrequency=500, bandPreset=False, harmonics=3)
        hpcp = hpcpAlg(freqs, mags)
        expected = [0., 0., 0., 0.1340538263, 0., 0.2476127148, 0., 0., 0., 0., 1., 0.]
        self.assertAlmostEqualVector(hpcp, expected, 1e-4)

    def testRegression(self):
        # Just makes sure algorithm does not crash on a real data source. This
        # test is not really looking for correctness. Maybe consider revising
        # it.
        inputSize = 512
        sampleRate = 44100

        audio = MonoLoader(filename = join(testdata.audio_dir, join('recorded', 'musicbox.wav')),
                           sampleRate = sampleRate)()

        fc = FrameCutter(frameSize = inputSize,
                         hopSize = inputSize)

        windowingAlg = Windowing(type = 'blackmanharris62')
        specAlg = Spectrum(size=inputSize)
        sPeaksAlg = SpectralPeaks(sampleRate = sampleRate,
                               maxFrequency = sampleRate/2,
                               minFrequency = 0,
                               orderBy = 'magnitude')

        hpcpAlg = HPCP(minFrequency=50, maxFrequency=500, bandPreset=False, harmonics=3)
        frame = fc(audio)
        while len(frame) != 0:
            spectrum = specAlg(windowingAlg(frame))
            (freqs, mags) = sPeaksAlg(spectrum)
            hpcp = hpcpAlg(freqs,mags)
            self.assertTrue(not any(numpy.isnan(hpcp)))
            self.assertTrue(not any(numpy.isinf(hpcp)))
            frame = fc(audio)


suite = allTests(TestHPCP)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
