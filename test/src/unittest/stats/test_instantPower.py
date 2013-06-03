#!/usr/bin/env python

from essentia_test import *

class TestInstantPower(TestCase):

    def testZero(self):
        self.assertEqual(InstantPower()(zeros(1000)), 0)

    def testEmptyOrOne(self):
        self.assertComputeFails(InstantPower(), [])
        self.assertEqual(InstantPower()([23]), 23*23)


    def testRegression(self):
        input = [0, 1, 2, 3]
        expected = 14./4.

        self.assertEqual(InstantPower()(input), expected)



suite = allTests(TestInstantPower)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

