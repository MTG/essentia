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

class TestVectorInput_Streaming(TestCase):

    def runNetwork(self, input):
        pool = Pool()
        gen = VectorInput(input)

        gen.data >> (pool, 'test')
        run(gen)
        return pool['test']

    def testVectorEmpty(self):
        pool = Pool()
        gen = VectorInput([])

        gen.data >> (pool, 'test')
        run(gen)
        self.assertEqualVector(pool.descriptorNames(), [])

    def testNotConnected(self):
        gen = VectorInput([1, 2, 3, 4, 5])
        self.assertRaises(EssentiaError, run, gen)

    def testVectorStereoSample(self):
        size = 10
        input = []
        for i in range(size):
            input.append([i, size - i])

        result = self.runNetwork(input)
        self.assertEqualMatrix(result, input)

    def testVectorInteger(self):
        input = [1, 2, 3, 4]
        result = self.runNetwork(input)
        self.assertEqualVector(result, input)

    def testVectorReal(self):
        input = [1.0, 2.0, 3.0, 4.0]
        result = self.runNetwork(input)
        self.assertEqualVector(result, input)

    def testString(self):
        input = ['foo', 'bar']
        result = self.runNetwork(input)
        self.assertEqualVector(result, input)

    def testMatrixReal(self):
        input = array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = self.runNetwork(input)
        self.assertEqualMatrix(result, input)

    def testMatrixReal2(self):
        input = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        result = self.runNetwork(input)
        self.assertEqualMatrix(result, input)

    def testMatrixInt(self):
        input = array([[1, 2, 3, 4], [5, 6, 7, 8]])
        result = self.runNetwork(input)
        self.assertEqualMatrix(result, input)

    def testMatrixInt2(self):
        input = [[1, 2, 3, 4], [5, 6, 7, 8]]
        result = self.runNetwork(input)
        self.assertEqualMatrix(result, input)

    #  still not supported
    #def testMatrixRealNotRectangular(self):
    #    input = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    #    result = self.runNetwork(input)
    #    self.assertEqualMatrix(result, input)

suite = allTests(TestVectorInput_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
