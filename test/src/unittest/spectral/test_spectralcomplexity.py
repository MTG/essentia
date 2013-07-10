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

class TestSpectralComplexity(TestCase):

    def testSinusoid(self):
        # make sure each frame's spectral complexity is zero since we only
        # have one frequency
        filename = join(testdata.audio_dir, 'generated/synthesised/sin5000.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()

        fc = FrameCutter(frameSize = 2048,
                         hopSize = 2048)

        windowing = Windowing(type = 'blackmanharris62')

        sc = SpectralComplexity(sampleRate = 44100,
                                magnitudeThreshold = .2)

        while True:
            frame = fc(audio)

            if len(frame) == 0:
                break

            spectrum = Spectrum()(windowing(frame))

            self.assertEqual(sc(spectrum), 1)

    def testZero(self):
        self.assertEqual(SpectralComplexity()([0]*100), 0)

    def testEmpty(self):
        self.assertComputeFails(SpectralComplexity(), [])

    def testOne(self):
        self.assertComputeFails(SpectralComplexity(), [0])

    def test10Impulses(self):
        # the first 100 zeros are padding to reach frequency 100Hz, which is
        # the minimum frequency used in spectralComplexity
        spectrum = [0]*100 + [0,1,10,1,0]*10
        sc = SpectralComplexity(magnitudeThreshold=9.9, sampleRate=len(spectrum)*2)
        self.assertEqual(sc(spectrum), 10)


suite = allTests(TestSpectralComplexity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
