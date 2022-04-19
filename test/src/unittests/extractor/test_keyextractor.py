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


from essentia_test import *
import numpy as np


class TestKeyExtractor(TestCase):
    def testEmpty(self):
        _, _, strength = KeyExtractor()(np.array([], dtype=np.float32))
        self.assertAlmostEqualFixedPrecision(strength, 0.0, 5)

    def testSilence(self):
        _, _, strength = KeyExtractor()(np.zeros(30 * 44100), dtype=np.float32)
        self.assertAlmostEqualFixedPrecision(strength, 0.0, 5)

    def testRegression(self):
        def test_on_real_audio(path, expected_key, expected_scale, strength_threshold):
            audio = MonoLoader(filename=path)()
            key, scale, strength = KeyExtractor()(audio)
            self.assertEqual(key, expected_key)
            self.assertEqual(scale, expected_scale)
            self.assertGreater(strength, strength_threshold)

        test_on_real_audio(join(testdata.audio_dir, "recorded", "mozart_c_major_30sec.wav"), 'C', 'major', 0.5)
        test_on_real_audio(join(testdata.audio_dir, "recorded", "Vivaldi_Sonata_5_II_Allegro.wav"), "E", "minor", 0.5)

suite = allTests(TestKeyExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)