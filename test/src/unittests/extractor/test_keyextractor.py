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
        _, _, strength = KeyExtractor()(np.zeros(30 * 44100, dtype=np.float32))
        self.assertAlmostEqualFixedPrecision(strength, 0.0, 5)

    def testRegression(self):
        def test_on_real_audio(path, expected_key, expected_scale, strength_threshold=0.5, wrong_answer_tolerance=2):
            profile_types = ["diatonic", "krumhansl", "temperley", "tonictriad", "temperley2005", "thpcp",
                             "shaath", "gomez", "noland", "edmm", "edma", "bgate", "braw"]
            correct_cnt = 0
            strengths = []
            for profile in profile_types:
                audio = MonoLoader(filename=path)()
                key, scale, strength = KeyExtractor(profileType=profile)(audio)
                if key == expected_key and scale == expected_scale:
                    correct_cnt += 1
                    strengths.append(strength)
                else:
                    strengths.append(-strength)

            self.assertGreaterEqual(correct_cnt, len(profile_types) - wrong_answer_tolerance)
            self.assertGreater(np.array(strengths).mean(), strength_threshold)

        test_on_real_audio(join(testdata.audio_dir, "recorded", "mozart_c_major_30sec.wav"), 'C', 'major', 0.5)
        test_on_real_audio(join(testdata.audio_dir, "recorded", "Vivaldi_Sonata_5_II_Allegro.wav"), "E", "minor", 0.5)

suite = allTests(TestKeyExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)