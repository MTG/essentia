#!/usr/bin/env python

from essentia_test import *

class TestFlatnessDB(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(FlatnessDB(), input)

    def testZeros(self):
        input = [0]*100
        result = FlatnessDB()(input)
        self.assertEqual(result, 1.0)

    def testFib(self):
        input = [1,1,2,3,5,8,13,21,34]
        result = FlatnessDB()(input)

        # calculated by hand (jk, with a calculator)
        expected = 0.047487197381536603376080461923790

        self.assertAlmostEqual(result, expected)


suite = allTests(TestFlatnessDB)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
