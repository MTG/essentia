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

class TestMedian(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Median(), [])

    def testZero(self):
        result = Median()([0]*10)
        self.assertEqual(result, 0)

    def testOne(self):
        result = Median()([100])
        self.assertEqual(result, 100)

    def testMulti(self):
        result = Median()([5, 8, 4, 9, 1])
        self.assertEqual(result, 5)

    def testNegatives(self):
        result = Median()([3, 7, -45, 2, -1, 0])
        self.assertEqual(result, 1)

    def testRational(self):
        result = Median()([3.1459, -0.4444, .00002])
        self.assertAlmostEqual(result, 0.00002)

    def testEvenSize(self):
        result = Median()([1, 4, 3, 10])
        self.assertAlmostEqual(result, 3.5)


suite = allTests(TestMedian)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
