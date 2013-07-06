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

maxfloat = 3.4028234663852886e+38

class TestOddToEvenHarmonicEnergyRatio(TestCase):
    def testEmpty(self):
        self.assertEqual(OddToEvenHarmonicEnergyRatio()([],[]), 1)

    def testOne(self):
        self.assertAlmostEqual(OddToEvenHarmonicEnergyRatio()([1], [1]), 0)

    def testZero(self):
        self.assertAlmostEqual(OddToEvenHarmonicEnergyRatio()([0], [0]), maxfloat )
        self.assertAlmostEqual(OddToEvenHarmonicEnergyRatio()([1], [0]), maxfloat )

    def testTwo(self):
        self.assertEqual(OddToEvenHarmonicEnergyRatio()([1,1], [1,1]), 1)

    def testDiffSizeInputs(self):
        self.assertComputeFails(OddToEvenHarmonicEnergyRatio(), [], [1])

    def testInputNotOrdered(self):
        self.assertComputeFails(
                OddToEvenHarmonicEnergyRatio(),
                [1, 3, 2], [1, 1, 1])

    def testZeroEvenEnergy(self):
        self.assertEqual(OddToEvenHarmonicEnergyRatio()([1,2,3,4], [0,1,0,1]), maxfloat)

    def testZeroOddEnergy(self):
        self.assertEqual(OddToEvenHarmonicEnergyRatio()([1,2,3,4], [1,0,1,0]), 0)

    def testRegression(self):
        mags = [1.23,4.32,1.22,6.23]
        evenEnergy = mags[0]**2 + mags[2]**2
        oddEnergy = mags[1]**2 + mags[3]**2
        self.assertAlmostEqual(
                OddToEvenHarmonicEnergyRatio()(range(len(mags)), mags),
                oddEnergy / evenEnergy,
                1e-6)


suite = allTests(TestOddToEvenHarmonicEnergyRatio)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
