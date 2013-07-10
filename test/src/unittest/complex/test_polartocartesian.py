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
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/


from essentia_test import *
from math import sin, cos

class TestPolarToCartesian(TestCase):

    def testEmpty(self):
        mags = []
        phases = []

        self.assertEqualVector(PolarToCartesian()(mags, phases), [])

    def testRegression(self):
        mags = [1, 4, 1.345, 0.321, -4]
        phases = [.45, 3.14, 2.543, 6.42, 1]

        expected = []
        for i in range(len(mags)):
            expected.append(mags[i] * cos(phases[i]) + (mags[i] * sin(phases[i]))*1j)

        self.assertAlmostEqualVector(PolarToCartesian()(mags, phases), expected, 1e-6)

    def testDiffSize(self):
        mags = [1]
        phases = [3, 4]
        self.assertComputeFails(PolarToCartesian(), mags, phases)


suite = allTests(TestPolarToCartesian)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
