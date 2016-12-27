#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
        f0 = 110
        freqs = [0.5, 0.75, 1.0, 2.0, 3.5, 4.0, 4.09, 4.9, 6.25]
        freqs = [freq*f0 for freq in freqs]
        mags = ones(len(freqs))
        result      = [1., 2., 3., 4., 4.9, 6., 7., 8., 9., 10. ]
        result_mags = [1,  1,  0,  1,  1,   0,  0,  0,  0,  0]      
        result_freqs = [freq*f0 for freq in result]
        
        f, m = HarmonicPeaks(maxHarmonics=10)(freqs, mags, f0)      
        self.assertAlmostEqualVector(f, result_freqs)
        self.assertEqualVector(m, result_mags)

        # in the case when two peaks are equidistant to an ideal harmonic
        # the peak with greater amplitude should be selected
        f0 = 110
        freqs = [1.,2.9, 3.1]
        mags = [1., 1.0, 0.5]
        freqs = [freq*f0 for freq in freqs]
        
        result = [1., 2., 2.9]
        result_mags = [1., 0., 1.]
        result_freqs = [freq*f0 for freq in result]

        f, m = HarmonicPeaks(maxHarmonics=3)(freqs, mags, f0)
        self.assertAlmostEqualVector(f, result_freqs)
        self.assertEqualVector(m, result_mags)

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

    def testMissingF0(self):
        f0 = 110
        freqs = [2.9, 3.1]
        mags = [1.0, 0.5]
        freqs = [freq*f0 for freq in freqs]
        
        f, m = HarmonicPeaks(maxHarmonics=3)(freqs, mags, f0)
        self.assertAlmostEqualVector(f, [110, 220, 319])
        self.assertEqualVector(m, [0, 0, 1])


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
        f, m = HarmonicPeaks(maxHarmonics=1)(freqs, mags, f0)
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
