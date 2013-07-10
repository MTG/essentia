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

class TestVariance(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Variance(), [])

    def testZero(self):
        result = Variance()([0]*10)
        self.assertAlmostEqual(result, 0)

    def testOne(self):
        result = Variance()([100])
        self.assertAlmostEqual(result, 0)

    def testMulti(self):
        result = Variance()([5, 8, 4, 9, 1])
        self.assertAlmostEqual(result, 8.24)

    def testNegatives(self):
        result = Variance()([3, 7, -45, 2, -1, 0])
        self.assertAlmostEqual(result, 315.888889)

    def testRational(self):
        result = Variance()([3.1459, -0.4444, .00002])
        self.assertAlmostEqual(result, 2.5538138)


suite = allTests(TestVariance)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
