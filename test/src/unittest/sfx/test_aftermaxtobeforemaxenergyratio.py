#!/usr/bin/env python

from essentia_test import *

class TestAfterMaxToBeforeMaxEnergyRatio(TestCase):

    def testEmpty(self):
        self.assertComputeFails(AfterMaxToBeforeMaxEnergyRatio(), [])

    def testZero(self):
        self.assertComputeFails(AfterMaxToBeforeMaxEnergyRatio(), [0]*100)

    def testOne(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()([1234]), 1)

    def testAscending(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()(range(10)) < 1, True)

    def testDescending(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()(range(9, -1, -1)) > 1, True)

    def testMaxInclusion(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()([100, 1000, 100]), 1)

    def testOnePitch(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()([100]), 1)


suite = allTests(TestAfterMaxToBeforeMaxEnergyRatio)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

