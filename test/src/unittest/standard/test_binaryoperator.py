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

class TestBinaryOperator(TestCase):

    div = BinaryOperator(type='/')
    prod = BinaryOperator(type='*')
    add = BinaryOperator(type='+')
    sub = BinaryOperator(type='-')

    def testRegression(self):
        input1 = [1, 2, 3, 4]
        input2 = [5, 6, 7, 8]
        self.assertEqualVector(self.add(input1, input2), [6, 8, 10, 12])
        self.assertEqualVector(self.sub(input1, input2), [-4, -4, -4, -4])
        self.assertEqualVector(self.prod(input1, input2),[5, 12, 21, 32])
        self.assertAlmostEqualVector(self.div(input1, input2), [1./5., 1./3., 3./7., 1./2.])

    def testZeroDivision(self):
        input1 = [1, 2, 3, 4]
        input2 = [0., 6, 7, 8]
        self.assertComputeFails(self.div, input1, input2)

    def testZero(self):
        input1 = zeros(10)
        input2 = zeros(10)
        self.assertEqualVector(self.add(input1, input2), input1)
        self.assertEqualVector(self.sub(input1, input2), input1)
        self.assertEqualVector(self.prod(input1, input2),input1)
        self.assertComputeFails(self.div, input1, input2)

    def testEmpty(self):
        input1 = []
        input2 = []
        self.assertEqualVector(self.add(input1, input2), [])
        self.assertEqualVector(self.sub(input1, input2), [])
        self.assertEqualVector(self.prod(input1, input2), [])
        self.assertEqualVector(self.div(input1, input2), [])


    def testInvalidParam(self):
        self.assertConfigureFails(BinaryOperator(), { 'type': '^'})

    def testDiffSize(self):
        input1 = [1]
        input2 = [4,5]
        self.assertComputeFails(self.add, input1, input2)

suite = allTests(TestBinaryOperator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
