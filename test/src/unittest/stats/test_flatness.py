#!/usr/bin/env python

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
