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


class TestHarmonicPeaks(TestCase):

    def testInvalidInput(self):
        # frequencies should be in ascendent order
        freqs = [440., 100., 660.]
        pitch = freqs[0]
        mags = ones(len(freqs))
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, pitch)
        # frequencies cannot be duplicated
        freqs = [440., 440., 660.]
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, pitch)
        # freqs and mags must have same size
        freqs = [100., 440., 660.]
        mags = ones(len(freqs)-1)
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, pitch)

    def testRegression(self):
        # frequencies should be in ascendent order
        f0 = 110
        freqs = [0.5, 0.75, 1.0, 2.0, 3.5, 4.0, 4.09, 4.9, 6]
        freqs = [freq*f0 for freq in freqs]
        result = [0.5, 1.0, 2.0, 4.0, 4.09]
        result = [freq*f0 for freq in result]
        mags = ones(len(freqs))
        f, m = HarmonicPeaks()(freqs, mags, f0)
        self.assertAlmostEqualVector(f, result)
        self.assertEqualVector(m, ones(len(result)))

        semitones = [0.5, 0.75, 12.5, 14.0, 23.9, 24.1, 24.9, 26, 32]
        freqs = [pow(2.0, s/12.0)*f0 for s in semitones]
        # freqs are [113.22324603078413, 114.87011606701552, 226.44649206156825,
        #            246.94165062806206, 437.46578647972075, 442.54889406985552,
        #            463.47885582012776, 493.88330125612401, 698.45646286600777]
        # in this case the fundamental closest to the given pitch is 113.223
        # obtaining the following real semitone relations:
        # [0.0, 0.25, 12.0, 13.5, 23.4, 23.6, 24.4, 25.5, 31.5]
        result = [0.5, 0.75, 12.5, 24.1, 24.9]
        result = [pow(2.0, s/12.0)*f0 for s in result]
        mags = ones(len(freqs))
        f, m = HarmonicPeaks()(freqs, mags, f0)
        self.assertAlmostEqualVector(f, result)
        self.assertAlmostEqualVector(m, ones(len(result)))

    def testNegativeFreqs(self):
        f0 = 110
        freqs = [-6, -4.9, -4.2, -4, -3.5, -2.0, -1.0, -0.75, -0.5]
        freqs = [freq*f0 for freq in freqs]
        mags = ones(len(freqs))
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, f0)

    def testNegativePitch(self):
        f0 = -110
        freqs = [-6, -4.9, -4.2, -4, -3.5, -2.0, -1.0, -0.75, -0.5]
        freqs = [freq*f0 for freq in freqs]
        mags = ones(len(freqs))
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, f0)

    def testEmpty(self):
        freqs = []
        f0 = 110
        mags = []

        hfreqs, hmags = HarmonicPeaks()(freqs, mags, f0)
        self.assertEqualVector(hfreqs, [])
        self.assertEqualVector(hmags, [])

    def testZero(self):
        freqs = zeros(10)
        f0 = 110
        mags = zeros(10)
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, f0)

    def testOnePeak(self):
        f0 = 110
        freqs = [f0]
        mags = [1]
        f, m = HarmonicPeaks()(freqs, mags, f0)
        self.assertEqualVector(f, freqs)
        self.assertEqualVector(m, mags)

    def testDC(self):
        f0 = 0
        freqs = [0, 110, 220]
        mags = [1, 0, 0]
        hfreqs, hmags = HarmonicPeaks()(freqs, mags, f0)
        self.assertEqualVector(hfreqs, [])
        self.assertEqualVector(hmags, [])

        f0 = 20
        freqs = [0, 40, 80]
        mags = [1, 0, 0]
        self.assertComputeFails(HarmonicPeaks(), freqs, mags, f0)


suite = allTests(TestHarmonicPeaks)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
