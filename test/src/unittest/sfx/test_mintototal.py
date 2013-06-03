#!/usr/bin/env python

from essentia_test import *

class TestMinToTotal(TestCase):

    def testEmpty(self):
        self.assertComputeFails(MinToTotal(), [])

    def testFlat(self):
        self.assertAlmostEquals(MinToTotal()([1]*100), 0)

    def testSingle(self):
        self.assertAlmostEquals(MinToTotal()([1]), 0)

    def testSimple(self):
        self.assertAlmostEquals(MinToTotal()([1,4,0.23,10.34]), 2/4.)


suite = allTests(TestMinToTotal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
