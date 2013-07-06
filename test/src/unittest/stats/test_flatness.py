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

class TestFlatness(TestCase):

    def testZero(self):
        self.assertEqual(Flatness()(zeros(1000)), 0)

    def testConstant(self):
        self.assertEqual(Flatness()([23]), 1)
        self.assertAlmostEqual(Flatness()([12]*237), 1, 1e-5)

    def testEmpty(self):
        self.assertComputeFails(Flatness(), [])

    def testNullValue(self):
        self.assertEqual(Flatness()([1, 2, 5, 3, 5.7, 0, 2, 6, 8 ]), 0)

    def testInvalidInput(self):
        self.assertComputeFails(Flatness(), [ 1, 2, 4, -3, 6, 7])

    def testRegression(self):
        inputArray = readVector(join(filedir(), 'flatness', 'input.txt'))
        flatness = readVector(join(filedir(), 'flatness', 'output.txt'))[0]

        self.assertAlmostEqual(Flatness()(inputArray), flatness, 1e-5)



suite = allTests(TestFlatness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
