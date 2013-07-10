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

class TestTristimulus(TestCase):

    def testZeroMag(self):
        mags = [0,0,0,0,0]
        freqs = [23, 500, 3200, 9000, 10000]

        self.assertEqualVector(
            Tristimulus()(freqs, mags),
            [0,0,0])


    def test3Freqs(self):
        mags = [1,2,3]
        freqs = [100, 200, 300]

        self.assertAlmostEqualVector(
            Tristimulus()(freqs, mags),
            [0.1666666667, 0, 0])


    def test4Freqs(self):
        mags = [1,2,3,4]
        freqs = [100, 435, 6547, 24324]

        self.assertAlmostEqualVector(
            Tristimulus()(freqs, mags),
            [.1, .9, 0])

    def test5Freqs(self):
        mags = [1,2,3,4,5]
        freqs = [100, 324, 5678, 5899, 60000]

        self.assertAlmostEqualVector(
            Tristimulus()(freqs, mags),
            [0.0666666667, .6, 0.33333333333])

    def testFrequencyOrder(self):
        freqs = [1,2,1.1]
        mags = [0,0,0]
        self.assertComputeFails(Tristimulus(), freqs, mags)

    def testFreqMagDiffSize(self):
        freqs = [1]
        mags = []
        self.assertComputeFails(Tristimulus(), freqs, mags)

    def testEmpty(self):
        freqs = []
        mags = []
        self.assertEqualVector(Tristimulus()([],[]), [0,0,0])


suite = allTests(TestTristimulus)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
