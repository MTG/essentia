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


class TestKey(TestCase):

    # This is a helper method that just runs the algorithm on a recorded wav.
    # One is able to specify the arguments passed to the Key algorithm.
    def runAlg(self, usePolyphony=True,
                     useThreeChords=True,
                     numHarmonics=4,
                     slope=0.6,
                     profileType='temperley',
                     pcpSize=36,
                     windowSize=0.5):

        sampleRate = 44100
        audio = MonoLoader(filename = join(testdata.audio_dir,
                                           join('recorded',
                                                'mozart_c_major_30sec.wav')),
                           sampleRate = sampleRate)()

        fc = FrameCutter(frameSize=4096, hopSize=512)

        windower = Windowing(type='blackmanharris62')

        specAlg = Spectrum(size=4096)
        sPeaksAlg = SpectralPeaks(sampleRate = sampleRate,
                                  maxFrequency = sampleRate/2,
                                  minFrequency = 0,
                                  orderBy = 'magnitude')

        hpcpAlg = HPCP(size=pcpSize, harmonics=numHarmonics-1, windowSize=0.5)

        # Calculate the average hpcp over all frames of audio
        frame = fc(audio)
        sums = [0]*pcpSize
        count = 0
        while len(frame) != 0:
            sPeaks = sPeaksAlg(specAlg(windower(frame)))
            hpcp = hpcpAlg(*sPeaks)

            for p in range(len(hpcp)):
                sums[p] += hpcp[p]

            count += 1
            frame = fc(audio)

        avgs = [x/count for x in sums]

        keyAlg = Key(usePolyphony=usePolyphony,
                     useThreeChords=useThreeChords,
                     numHarmonics=numHarmonics,
                     slope=slope,
                     profileType=profileType,
                     pcpSize=pcpSize)

        return keyAlg(avgs)

    def assertValidSequence(self, s):
        keys = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
        scales = ['major', 'minor']

        self.assertTrue(s[0] in keys)
        self.assertTrue(s[1] in scales)
        self.assertTrue(not numpy.isnan(s[2]))
        self.assertTrue(not numpy.isnan(s[3]))
        self.assertTrue(not numpy.isinf(s[2]))
        self.assertTrue(not numpy.isinf(s[3]))



    # The following tests just run the runAlg function. With the exception of
    # testRegression, these tests are not looking for accurate output, they are
    # just testing whether the program doesn't crash when different parameters
    # are given.
    def testRegression(self):
        (key, scale, strength, firstToSecondRelativeStrength) = self.runAlg()
        self.assertEqual(key, 'C')
        self.assertEqual(scale, 'major')
        self.assertAlmostEqual(strength, 0.760322451591, 1e-6)
        self.assertAlmostEqual(firstToSecondRelativeStrength, 0.607807099819, 1e-6)

    def testUsePolyphonyFalse(self):
        self.assertValidSequence(self.runAlg(usePolyphony=False))

    def testThreeChordsFalse(self):
        self.assertValidSequence(self.runAlg(useThreeChords=False))

    def testVariousProfileTypes(self):
        import sys
        sys.stdout.write('Testing diatonic')
        sys.stdout.flush()
        self.assertValidSequence(self.runAlg(profileType='diatonic'))
        sys.stdout.write(', krumhansl')
        sys.stdout.flush()
        self.assertValidSequence(self.runAlg(profileType='krumhansl'))
        sys.stdout.write(', weichai')
        sys.stdout.flush()
        self.assertValidSequence(self.runAlg(profileType='weichai'))
        sys.stdout.write(', tonictriad')
        sys.stdout.flush()
        self.assertValidSequence(self.runAlg(profileType='tonictriad'))
        sys.stdout.write(', temperley2005')
        sys.stdout.flush()
        self.assertValidSequence(self.runAlg(profileType='temperley2005'))
        sys.stdout.write(', thpcp ... ')
        sys.stdout.flush()
        self.assertValidSequence(self.runAlg(profileType='thpcp'))

    def testNumHarmonics(self):
        self.assertValidSequence(self.runAlg(numHarmonics=1))


suite = allTests(TestKey)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
