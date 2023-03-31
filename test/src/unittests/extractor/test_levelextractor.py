#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

class TestLevelExtractor(TestCase):

    def testEmpty(self):
        empty_vec = np.array([], dtype=np.single)
        self.assertRaises(RuntimeError, lambda: LevelExtractor()(empty_vec))
        # TODO: Actual errmsg should be more readable.

    def testSilence(self):
        silence_vec = np.zeros(44100, dtype=np.single)
        res = LevelExtractor()(silence_vec)
        self.assertEqual(len(res), 1)
        self.assertAlmostEqualFixedPrecision(res[0], 0, 5)

    def testInvalidParam(self):
        self.assertConfigureFails(LevelExtractor(), {'frameSize': -1, 'hopSize': -1})
        self.assertConfigureFails(LevelExtractor(), {'frameSize': -1, 'hopSize': 44100})
        self.assertConfigureFails(LevelExtractor(), {'frameSize': 88200, 'hopSize': -1})
        self.assertConfigureFails(LevelExtractor(), {'frameSize': 0, 'hopSize': 0})
        self.assertConfigureFails(LevelExtractor(), {'frameSize': 88200, 'hopSize': 0})

        self.assertConfigureSuccess(LevelExtractor(), {'frameSize': 44100, 'hopSize': 22050})

    def testRandomInput(self):
        n = 10
        for _ in range(n):
            rand_input = np.random.random(88200).astype(np.single) * 2 - 1
            res = LevelExtractor()(rand_input)
            expected_res = np.sum(rand_input * rand_input) ** 0.67
            self.assertAlmostEqual(res[0], expected_res, 1e-4)

    def testRealAudio(self):
        input_filename = join(testdata.audio_dir, "recorded", "dubstep.wav")
        audio = MonoLoader(filename=input_filename)()

        def test(frameSize, hopSize, assertion=self.assertAlmostEqualVector, error=1e-4):
            def ground_truth():
                res = []
                for i in range(0, len(audio) - frameSize + hopSize, hopSize):
                    res.append(np.sum(np.square(audio[i: i+frameSize])) ** 0.67)
                return np.array(res, dtype=np.single)

            def get_extractor_result():
                le = LevelExtractor()
                le.configure(frameSize=frameSize, hopSize=hopSize)
                return le(audio)

            assertion(get_extractor_result(), ground_truth(), error)

        test(44100, 22050)
        test(88200, 44100)
        test(88200, 9999)
        test(2, 1)
        test(512, 128)
        test(44100 * 4, 44100)



suite = allTests(TestLevelExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
