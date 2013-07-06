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

class TestSpectralPeaks(TestCase):

    def testRegression(self):
        # NB: the schemas of the peak types don't look correct here...
        spectrum = [0] * 100

        # \___
        spectrum[0:3]   = [ 0.5, 0.4, 0.3 ]

        # __/\__
        spectrum[10:13] = [ 0.5, 0.6, 0.5 ]

        # __/`---\__
        spectrum[20:25] = [ 0.8, 0.95, 0.95, 0.95, 0.8 ]

        # __/---\__
        spectrum[30:35] = [ 0.5, 0.6, 0.6, 0.6, 0.7 ]

        # __/`--\__
        spectrum[40:44] = [ 0.5, 0.6, 0.6, 0.7 ]

        # __/`---\__
        spectrum[50:55] = [ 0.7, 0.6, 0.6, 0.6, 0.5 ]

        # __/`---\__
        spectrum[60:65] = [ 0.7, 0.6, 0.6, 0.6, 0.7 ]

        # __/`\__
        spectrum[70:75] = [ 0.7, 0.5, 0.7, 0.5, 0.7 ]

        # ___/
        spectrum[97:100] = [ 0.3, 0.4, 0.5 ]

        speaks = SpectralPeaks(sampleRate = 99*2,
                               maxPeaks = 100,
                               maxFrequency = 99,
                               minFrequency = 0,
                               magnitudeThreshold = 1e-6,
                               orderBy = 'frequency')

        (freqs, mags) = speaks(spectrum)
        peaks = zip(freqs, mags)

        expected = [ (0, 0.5),
                     (11, 0.6),
                     (22, 0.95),
                     (33.625, 0.75625),
                     (42.625, 0.75625),
                     (50.375, 0.75625),
                     (60.375, 0.75625),
                     (63.625, 0.75625),
                     (70.2778, 0.734722),
                     (72, 0.7),
                     (73.7222, 0.734722),
                     (99, 0.5)
                     ]

        for exp, found in zip(expected, peaks):
            self.assertAlmostEqual(found[0], exp[0], 1e-6)
            self.assertAlmostEqual(found[1], exp[1], 1e-6)

    def testSinusoid(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin5000.wav'),
                           sampleRate = 44100)()

        fc = FrameCutter(frameSize = 2048,
                         hopSize = 2048)

        windowing = Windowing(type = 'blackmanharris62')
        spec = Spectrum()

        speaks = SpectralPeaks(sampleRate = 44100,
                               maxPeaks = 1,
                               maxFrequency = 10000,
                               minFrequency = 0,
                               magnitudeThreshold = 0,
                               orderBy = 'magnitude')

        while True:
            frame = fc(audio)

            if len(frame) == 0:
                break

            (freqs, mags) = speaks(spec(windowing(frame)))

            self.assertAlmostEqual(freqs[0], 5000, 1e-3)

    def testZero(self):
        speaks = SpectralPeaks()
        (freqs, mags) = speaks(numpy.zeros(1024, dtype='f4'))
        self.assert_(len(freqs) == 0)
        self.assert_(len(mags) == 0)

    def testEmpty(self):
        # Feeding an empty array shouldn't crash and throw an exception
        self.assertComputeFails(SpectralPeaks(), [])

    def testOne(self):
        # Feeding an array of size 1 shouldn't crash and throw an exception
        self.assertComputeFails(SpectralPeaks(), [0])

    def testInvalidParam(self):
        self.assertConfigureFails(SpectralPeaks(), {'sampleRate': 0})
        self.assertConfigureFails(SpectralPeaks(), {'maxPeaks': 0})
        self.assertConfigureFails(SpectralPeaks(), {'maxFrequency': 0})
        self.assertConfigureFails(SpectralPeaks(), {'minFrequency': -1})
        self.assertConfigureFails(SpectralPeaks(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(SpectralPeaks(), {'orderBy': 'Unknown'})
        self.assertConfigureFails(SpectralPeaks(), {'minFrequency': 100, 'maxFrequency': 99})

suite = allTests(TestSpectralPeaks)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
