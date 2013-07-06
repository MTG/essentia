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


class TestDanceability(TestCase):
    def testEmpty(self):
        self.assertEqual(Danceability()([]), 0)

    def testOne(self):
        self.assertEqual(Danceability()([10]), 0)

    def testSilence(self):
        self.assertEqual(Danceability()([0]*44100), 0)

    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded', 'roxette.wav'),
                            downmix='left', sampleRate=44100)()
        self.assertAlmostEqual(Danceability()(audio), 1.25313031673, 5e-3)

    def testSanity(self):
        filename = join(testdata.audio_dir, 'recorded', \
                        '01-Allegro__Gloria_in_excelsis_Deo_in_D_Major.wav')

        gloriaAudio = MonoLoader(filename=filename,
                                 downmix='left',
                                 sampleRate=44100)()
        filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        britneyAudio = MonoLoader(filename=filename,
                                  downmix='left',
                                  sampleRate=44100)()

        gloriaDanceability = Danceability()(gloriaAudio)
        britneyDanceability = Danceability()(britneyAudio)

        self.assertTrue(gloriaDanceability < britneyDanceability)

    def testMinBiggerThanMaxTau(self):
        self.assertConfigureFails(Danceability(), {'minTau':1000, 'maxTau':500})

    def testZero(self):
        input = [0]*100000

        self.assertAlmostEqual(Danceability()(input), 0, 1e-2)


suite = allTests(TestDanceability)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
