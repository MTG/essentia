#!/usr/bin/env python

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
