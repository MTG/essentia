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

class TestResampleFFT(TestCase):

    def testEmpty(self):
        empty = np.array([], dtype=np.single)
        r = ResampleFFT()
        with self.assertRaises(RuntimeError):
            r(empty)

    def testSilence(self):
        silence = np.zeros(44100, dtype=np.single)
        r = ResampleFFT()
        r.configure(outSize=10492 * 3)  # Arbitrary value used
        output = r(silence)
        self.assertAlmostEqualVectorFixedPrecision(output, np.zeros(10492 * 3, dtype=np.single), 6)

    def testConstant(self):
        # Comparing with the results generated from scipy.signal.resample
        ones = np.ones(114, dtype=np.single)
        r = ResampleFFT()
        out = r(ones)
        self.assertAlmostEqualVector(out, np.ones(128, dtype=np.single), 1e-5)
        r.reset()
        out = r(np.full(114, -1, dtype=np.single))
        self.assertAlmostEqualVector(out, np.full(128, -1, dtype=np.single), 1e-5)

    def testRecordedAudio(self):
        # Test data is from real_audio PCM, and the expected result is precalculated with scipy.signal.resample
        # Due to the nature of FFT resample, it is improper to test with random input.
        import os.path as path
        import pathlib
        with np.load(path.join(pathlib.Path(__file__).resolve().parent, "resamplefft", "dubstep_first_second.npz")) as data:
            raw = data["raw"]
            expected = data["expected"]
            r = ResampleFFT()
            r.configure(inSize=44100, outSize=32768)
            output = r(raw)
            self.assertAlmostEqualVectorFixedPrecision(output, expected, 5)  # The error is a little bit too big


    def testOddLength(self):
        import os.path as path
        import pathlib
        with np.load(path.join(pathlib.Path(__file__).resolve().parent, "resamplefft", "mozart_c_major_fragment.npz")) as data:
            raw = data["raw"]
            expected = data["expected"]
            r = ResampleFFT()
            r.configure(inSize=110249, outSize=100001)
            output = r(raw)
            self.assertAlmostEqualVectorFixedPrecision(output, expected, 5)

suite = allTests(TestResampleFFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
