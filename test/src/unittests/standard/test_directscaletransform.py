#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

testdir = join(filedir(), 'directscaletransform')

class TestDirectScaleTransform(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(DirectScaleTransform(), { 'C': -1, 'fs': -2 })

    def testIncorrectInputs(self):
        self.assertComputeFails(DirectScaleTransform(), []) # testEmpty
        self.assertComputeFails(DirectScaleTransform(), [23]) # input should be 2D vector

    def testZero(self):
        self.assertEqualMatrix(DirectScaleTransform()(zeros([20,10])), zeros([485, 10]))    # 500/(M_PI/ln(20+1))+1 = 485.55

    def testRegression(self):
        inputMatrix = readMatrix(join(testdir, 'input.txt'))
        outputMatrix = readMatrix(join(testdir, 'output.txt'))
        dst = DirectScaleTransform()(inputMatrix)
        self.assertAlmostEqualMatrix(dst, outputMatrix, 1e-2)


suite = allTests(TestDirectScaleTransform)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

