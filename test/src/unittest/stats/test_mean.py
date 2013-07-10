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

class TestMean(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Mean(), [])

    def testZero(self):
        result = Mean()([0]*10)
        self.assertAlmostEqual(result, 0)

    def testOne(self):
        result = Mean()([100])
        self.assertAlmostEqual(result, 100)

    def testMulti(self):
        result = Mean()([5, 8, 4, 9, 1])
        self.assertAlmostEqual(result, 5.4)

    def testNegatives(self):
        result = Mean()([3, 7, -45, 2, -1, 0])
        self.assertAlmostEqual(result, -5.666666666)

    def testRational(self):
        result = Mean()([3.1459, -0.4444, .00002])
        self.assertAlmostEqual(result, 0.900506666667)


suite = allTests(TestMean)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
