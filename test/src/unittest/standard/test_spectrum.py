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

testdir = join(filedir(), 'spectrum')


class TestSpectrum(TestCase):

    def testRegression(self):
        input = readVector(join(testdir, 'input.txt'))
        expected = readVector(join(testdir, 'output.txt'))
        output = Spectrum()(input)
        self.assertAlmostEqualVector(expected, output, 1e-4)

    def testDC(self):
        inputSize = 512
        signalDC = [1] * inputSize
        expectedDC = [0] * int(inputSize / 2 + 1)
        expectedDC[0] = inputSize
        outputDC = Spectrum()(signalDC)
        self.assertEqualVector(outputDC,  expectedDC)

    def testNyquist(self):
        inputSize = 512
        signalNyquist = [-1,  1] * (inputSize / 2)
        expectedNyquist = [0] * int(inputSize / 2 + 1)
        expectedNyquist[-1] = inputSize
        outputNyquist = Spectrum()(signalNyquist)
        self.assertEqualVector(outputNyquist,  expectedNyquist)

    def testZero(self):
        inputSize = 512
        signalZero = [0] * inputSize
        expectedZero = [0] * int(inputSize / 2 + 1)
        outputZero = Spectrum()(signalZero)
        self.assertEqualVector(outputZero,  expectedZero)

    def testSize(self):
        inputSize = 512
        fakeSize = 514
        expectedSize = int(inputSize / 2 + 1)
        input = [1] * inputSize
        output = Spectrum(size=fakeSize)(input)
        self.assertEqual(len(output), expectedSize)

    def testEmpty(self):
        # Checks whether an empty input vector yields an exception
        self.assertComputeFails(Spectrum(),  [])

    def testOne(self):
        # Checks for a single value
        #self.assertEqual(Spectrum()([1]),  [1])
        self.assertComputeFails(Spectrum(),  [1])

    def testInvalidParam(self):
        self.assertConfigureFails(Spectrum(), {'size': -1})
        self.assertConfigureFails(Spectrum(), {'size': 0})

suite = allTests(TestSpectrum)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
