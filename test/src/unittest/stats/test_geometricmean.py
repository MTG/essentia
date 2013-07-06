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

class TestGeometricMean(TestCase):

    def testZero(self):
        self.assertEqual(GeometricMean()(zeros(1000)), 0)

    def testEmpty(self):
        self.assertComputeFails(GeometricMean(), [])

    def testNullValue(self):
        self.assertEqual(GeometricMean()([1, 23, 46, 2, 6, 13, 0, 35, 57, 3]), 0)

    def testInvalidInput(self):
        self.assertComputeFails(GeometricMean(), [ 1, 5, 3, -7, 2, 67 ])

    def testRegression(self):
        self.assertAlmostEqual(GeometricMean()([2]), 2)
        self.assertAlmostEqual(GeometricMean()([0.25, 0.5, 1]), 0.5)
        self.assertAlmostEqual(GeometricMean()([32, 16, 2, 4, 8]), 8, 2e-7)



suite = allTests(TestGeometricMean)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

