#!/usr/bin/env python

from essentia_test import *

class TestDecrease(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Decrease(), [])

    def testZero(self):
        self.assertEqual(Decrease(range=1.0)([0]*666), 0)

    def testOne(self):
        self.assertComputeFails(Decrease(), [666])

    def testRegression(self):
        self.assertEqual(Decrease(range=4)(range(1, 6)), 1)


suite = allTests(TestDecrease)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
