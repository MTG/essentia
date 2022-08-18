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


class TestTuningFrequencyExtractor(TestCase):
    def testEmpty(self):
        with self.assertRaises(RuntimeError):
            est = TuningFrequencyExtractor()(np.array([], dtype=np.float32))

    def testSilence(self):
        # The algorithm always produces output on any non-empty input audio.
        # It applies a magnitude thresholding for peak detection. Complete
        # silence gets some noise added by FrameCutter with peak magnitudes
        # below this threshold. Therefore, the output for the constant-valued
        # input varies: 440 when there are no peaks above the threshold and
        # 434.1931 otherwise.

        self.assertAlmostEqualAbs(TuningFrequencyExtractor()(
            np.zeros((88200,), dtype=np.float32))[-1], 440.0, 4.0)
        self.assertAlmostEqualAbs(TuningFrequencyExtractor()(
            np.ones((88200,), dtype=np.float32))[-1], 434.1931, 4.0)

    def testInvalidParam(self):
        self.assertConfigureFails(TuningFrequencyExtractor(), {'frameSize': -1, 'hopSize': -1})
        self.assertConfigureFails(TuningFrequencyExtractor(), {'frameSize': -1, 'hopSize': 1024})
        self.assertConfigureFails(TuningFrequencyExtractor(), {'frameSize': 8192, 'hopSize': -1})
        self.assertConfigureFails(TuningFrequencyExtractor(), {'frameSize': 0, 'hopSize': 0})
        self.assertConfigureFails(TuningFrequencyExtractor(), {'frameSize': 8192, 'hopSize': 0})

        self.assertConfigureSuccess(TuningFrequencyExtractor(), {'frameSize': 8192, 'hopSize': 1024})

    def testSynthesized(self):
        audio_219hz = \
            MonoLoader(filename=join(testdata.audio_dir, "generated", "synthesised", "sine_219hz_1sec_0db.wav"))()
        # self.assertAlmostEqualFixedPrecision(TuningFrequencyExtractor()(audio_219hz)[-1], 438.0, 1)
        self.assertAlmostEqualAbs(TuningFrequencyExtractor(frameSize=16384, hopSize=4096)(audio_219hz)[-1], 438.0, 0.5)

        # Test on weird frame size selection.
        self.assertAlmostEqualAbs(TuningFrequencyExtractor(frameSize=10000, hopSize=2000)(audio_219hz)[-1], 438.0, 1.0)

        # When we using a shorter frameSize, the frequency resolution will drop, so the result could be more inaccurate
        self.assertAlmostEqualAbs(TuningFrequencyExtractor(frameSize=4096, hopSize=2048)(audio_219hz)[-1], 438.0, 2.0)

        audio_440hz = MonoLoader(filename=join(testdata.audio_dir, "generated", "synthesised", "sin440_0db.wav"))()
        self.assertAlmostEqualAbs(TuningFrequencyExtractor()(audio_440hz)[-1], 440.0, 1.0)
        self.assertAlmostEqualAbs(TuningFrequencyExtractor(frameSize=2048, hopSize=1024)(audio_440hz)[-1], 440.0, 1.0)

    def testReal(self):
        guitar_audio_1 = MonoLoader(filename=join(testdata.audio_dir, "recorded", "Guitar-A4-432-1.ogg"))()
        guitar_audio_2 = MonoLoader(filename=join(testdata.audio_dir, "recorded", "Guitar-A4-432-2.ogg"))()

        self.assertAlmostEqualAbs(TuningFrequencyExtractor()(guitar_audio_1)[-1], 432.0, 2.0)
        self.assertAlmostEqualAbs(TuningFrequencyExtractor(frameSize=8192)(guitar_audio_1)[-1], 432.0, 1.0)
        self.assertAlmostEqualAbs(TuningFrequencyExtractor()(guitar_audio_2)[-1], 432.0, 2.0)
        self.assertAlmostEqualAbs(TuningFrequencyExtractor(frameSize=8192)(guitar_audio_2)[-1], 432.0, 1.0)


suite = allTests(TestTuningFrequencyExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
