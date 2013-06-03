#!/usr/bin/env python

from essentia_test import *


class TestDCT(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(DCT(), { 'inputSize': 0, 'outputSize': 2 })
        self.assertConfigureFails(DCT(), { 'inputSize': 6, 'outputSize': 0 })


    def testRegression(self):
        # values from Matlab/Octave
        inputArray = [ 0, 0, 1, 0, 1 ]
        expected = [ 0.89442719099, -0.60150095500, -0.12078825843, -0.37174803446, 0.82789503961 ]
        self.assertAlmostEqualVector(DCT(outputSize=len(inputArray))(inputArray), expected, 1e-6)


    def testZero(self):
        self.assertEqualVector(DCT(outputSize=10)(zeros(20)), zeros(10))

    def testInvalidInput(self):
        self.assertComputeFails(DCT(), []) # = testEmpty
        self.assertComputeFails(DCT(outputSize = 10), [ 0, 2, 4 ])




suite = allTests(TestDCT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
