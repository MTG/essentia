#!/usr/bin/env python

from essentia_test import *

class TestCrest(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Crest(), [])

    def testZero(self):
        self.assertEqual(Crest()([0]*666), 0)

    def testOne(self):
        self.assertEqual(Crest()([666]), 1)

    def testNegative(self):
        self.assertComputeFails(Crest(), [-1]*666)


suite = allTests(TestCrest)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
