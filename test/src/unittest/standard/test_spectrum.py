#!/usr/bin/env python

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
