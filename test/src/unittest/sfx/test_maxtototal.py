#!/usr/bin/env python

from essentia_test import *

class TestMaxToTotal(TestCase):

    def testEmpty(self):
        self.assertComputeFails(MaxToTotal(), [])

    def testFlat(self):
        self.assertAlmostEquals(MaxToTotal()([1]*100), 0)

    def testSingle(self):
        self.assertAlmostEquals(MaxToTotal()([1]), 0)

    def testSimple(self):
        self.assertAlmostEquals(MaxToTotal()([1,4,0.23,10.34]), 3/4.)


suite = allTests(TestMaxToTotal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
