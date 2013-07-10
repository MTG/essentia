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

class TestScale(TestCase):

    def testRegression(self):
        inputSize = 1024
        input = range(inputSize)
        factor = 0.5
        expected = [factor * n for n in input]
        output = Scale(factor=factor, clipping=False)(input)
        self.assertEqualVector(output, expected)

    def testZero(self):
        inputSize = 1024
        input = [0] * inputSize
        expected = input[:]
        output = Scale()(input)
        self.assertEqualVector(output, input)

    def testEmpty(self):
        input = []
        expected = input[:]
        output = Scale()(input)
        self.assertEqualVector(output, input)

    def testClipping(self):
        inputSize = 1024
        maxAbsValue= 10
        factor = 1
        input = [n + maxAbsValue for n in range(inputSize)]
        expected = [maxAbsValue] * inputSize
        output = Scale(factor=factor, clipping=True, maxAbsValue=maxAbsValue)(input)
        self.assertEqualVector(output, expected)

    def testInvalidParam(self):
        self.assertConfigureFails(Scale(), { 'maxAbsValue': -1 })

suite = allTests(TestScale)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
