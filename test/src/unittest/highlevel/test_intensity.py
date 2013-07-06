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


class TestIntensity(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Intensity(), [])

    def testSilence(self):
        audio = [0]*(44100*10) # 10 sec silence
        self.assertEqual(Intensity()(audio), -1) # silence is relaxing isn't it

    def testDiffSampleRates(self):
        algo44 = Intensity(sampleRate=44100)
        algo22 = Intensity(sampleRate=22050)

        filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        britney44 = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        filename = join(testdata.audio_dir, 'recorded', 'britney22050.wav')
        britney22 = MonoLoader(filename=filename, downmix='left', sampleRate=22050)()

        self.assertEqual(algo44(britney44), algo22(britney22))

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        britneyIntensity = Intensity()(audio)

        filename = join(testdata.audio_dir, 'recorded', '01-Allegro__Gloria_in_excelsis_Deo_in_D_Major.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        gloriaIntensity = Intensity()(audio)

        filename = join(testdata.audio_dir, 'recorded', 'roxette.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        roxetteIntensity = Intensity()(audio)

        self.assertTrue(britneyIntensity > gloriaIntensity)
        self.assertTrue(britneyIntensity >= roxetteIntensity)
        self.assertTrue(roxetteIntensity > gloriaIntensity)


suite = allTests(TestIntensity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
