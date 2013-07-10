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



import math
from essentia_test import *


class TestStrongPeak(TestCase):

    def testEmpty(self):
        # strong peak must be bigger than one element
        self.assertComputeFails(StrongPeak(), [])

    def testOne(self):
        # strong peak must be bigger than one element
        self.assertComputeFails(StrongPeak(), [1])

    def testFlat(self):
        # strongpeak not defined when no peak exists
        self.assertEqual(StrongPeak()([1]*100), 0)

    def testImpulse(self):
        self.assertAlmostEqual(StrongPeak()([0]*50 + [1] + [0]*50), 1/math.log10(51/50.), 1e-6)

    def testNegatives(self):
        # strongpeak ratio not defined for negative spectrums
        self.assertComputeFails(StrongPeak(), [-10]*50 + [10]*50)

    def testZero(self):
        # strongpeak not defined when no peak exists
        self.assertEqual(StrongPeak()([0]*100), 0)

    def testManyPeaks(self):
        self.assertAlmostEqual(StrongPeak()([0,0,0,.3,0,0,0,.9,0,0,0,1,0,0,0]), 1 / math.log10(12/11.), 5e-7)

    def testEqualPeaks(self):
        self.assertAlmostEqual(StrongPeak()([0,0,0,1,0,0,1]), 1 / math.log10(4/3.), 5e-7)

    def testSquarePeak(self):
        self.assertAlmostEqual(StrongPeak()([0,0,0,3,3,3,0,0,0]), 3 / math.log10(6/3.))

    def testSimple(self):
        self.assertAlmostEqual(StrongPeak()([0,1,4,5,9,5,4,1,0]), 9 / math.log10(2))


suite = allTests(TestStrongPeak)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
