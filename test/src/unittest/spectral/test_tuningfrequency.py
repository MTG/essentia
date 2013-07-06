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

class TestTuningFrequency(TestCase):

    def testSinusoid(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_0db.wav'),
                           sampleRate = 44100)()

        fc = FrameCutter(frameSize = 2048,
                         hopSize = 2048)


        windowing = Windowing(type = 'hann')
        spec = Spectrum()

        speaks = SpectralPeaks(sampleRate = 44100,
                               maxPeaks = 1,
                               maxFrequency = 10000,
                               minFrequency = 0,
                               magnitudeThreshold = 0,
                               orderBy = 'magnitude')

        tfreq = TuningFrequency(resolution=1.0)

        while True:
            frame = fc(audio)

            if len(frame) == 0:
                break

            (freqs, mags) = speaks(spec(windowing(frame)))
            (freq, cents) = tfreq(freqs,  mags)

            self.assertAlmostEqual(cents, 0, 4)
            self.assertAlmostEqual(freq, 440.0, 0.5)

    def testEmpty(self):
        (freq,  cents) = TuningFrequency()([],  [])
        self.assertAlmostEqual(freq, 440)
        self.assertAlmostEqual(cents, 0)

    def testZero(self):
        (freq,  cents) = TuningFrequency()([0],  [0])
        self.assertAlmostEqual(freq, 440)
        self.assertAlmostEqual(cents, 0)

    def testSizes(self):
        # Checks whether an exception gets raised,
        # in case the frequency vector has a different size than the magnitude vector
        self.assertComputeFails(TuningFrequency(),  [1],  [1,  2])

    def testInvalidParam(self):
        self.assertConfigureFails(TuningFrequency(), { 'resolution': 0 })

suite = allTests(TestTuningFrequency)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
