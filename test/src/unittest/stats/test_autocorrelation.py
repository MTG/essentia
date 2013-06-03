#!/usr/bin/env python

from essentia_test import *

testdir = join(filedir(), 'autocorrelation')


class TestAutoCorrelation(TestCase):

    def testRegression(self):
        inputv = readVector(join(testdir, 'input_pow2.txt'))
        expected = readVector(join(testdir, 'output.txt'))

        output = AutoCorrelation()(inputv)

        self.assertAlmostEqualVector(expected, output, 1e-4)


    def testNonPowerOfTwo(self):
        inputv = readVector(join(testdir, 'octave_input.txt'))
        inputv = inputv[:234]
        expected = readVector(join(testdir, 'output_nonpow2.txt'))

        output = AutoCorrelation()(inputv)

        self.assertAlmostEqualVector(expected, output, 1e-4)


    def testOctave(self):
        inputv = readVector(join(testdir, 'octave_input.txt'))
        expected = readVector(join(testdir, 'octave_output.txt'))

        output = AutoCorrelation()(inputv)

        self.assertEqual(len(expected)/2, len(output))

        self.assertAlmostEqualVector(expected[:len(expected)/2], output, 1e-4)


    def testZero(self):
        self.assertEqualVector(AutoCorrelation()(zeros(1024)), zeros(1024))

    def testEmpty(self):
        self.assertEqualVector(AutoCorrelation()([]), [])

    def testOne(self):
        self.assertAlmostEqualVector(AutoCorrelation()([0.2]), [0.04])

    def testInvalidParam(self):
        self.assertConfigureFails(AutoCorrelation(), { 'normalization': 'unknown' })


suite = allTests(TestAutoCorrelation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
