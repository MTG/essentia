#!/usr/bin/env python

# Copyright (C) 2006-2022  Music Technology Group - Universitat Pompeu Fabra
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
import os.path

from essentia_test import *
import numpy as np


class TestKeyExtractor(TestCase):
    def testEmpty(self):
        with self.assertRaises(RuntimeError):
            _, _, strength = KeyExtractor()(np.array([], dtype=np.float32))

    def testSilence(self):
        _, _, strength = KeyExtractor()(np.zeros(30 * 44100, dtype=np.float32))
        self.assertAlmostEqualFixedPrecision(strength, 0.0, 5)

    def testRegression(self):
        profile_types = ["diatonic", "krumhansl", "temperley", "temperley2005", "thpcp", "shaath", "gomez", "noland",
                         "edmm", "edma", "bgate", "braw"]
        print()

        def test_on_real_audio(path, expected_key, expected_scale, profiles=profile_types, strength_threshold=0.5):
            audio = MonoLoader(filename=path)()
            print(f"Test on file {os.path.basename(path)}")

            if type(strength_threshold) is float:
                strength_threshold = [strength_threshold for p in profiles]
            elif type(strength_threshold) != list:
                raise TypeError("Unsupported type of profile")

            for profile, thres in zip(profiles, strength_threshold):
                key, scale, strength = KeyExtractor(profileType=profile)(audio)
                print(profile, key, scale, strength)
                self.assertEqual(key, expected_key)
                self.assertEqual(scale, expected_scale)
                self.assertGreater(strength, thres)

        test_on_real_audio(join(testdata.audio_dir, "recorded", "mozart_c_major_30sec.wav"), 'C', 'major',
                           ['temperley2005', 'temperley'], 0.7)
        test_on_real_audio(join(testdata.audio_dir, "recorded", "Vivaldi_Sonata_5_II_Allegro.wav"), "E", "minor",
                           ['temperley2005', 'temperley'], 0.7)

suite = allTests(TestKeyExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)