#!/usr/bin/env python

from essentia_test import *

class TestNoiseAdder(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(NoiseAdder(), { 'level': 10 })

    def testLevel(self):
        ng = NoiseAdder(level = 0)
        level_0 = ng(zeros(1000))
        self.assert_(all(abs(level_0) <= 1))

        ng.configure(level = -10)
        level_0_1 = ng(zeros(1000))
        k = db2lin(-10.0)
        self.assert_(all(abs(level_0_1) <= k))

        ng.configure(level = -30)
        level_0_3 = ng(ones(1000))
        k = db2lin(-30.0)
        self.assert_(all(abs(level_0_3 - ones(1000)) <= k))

    def testEmpty(self):
        self.assertEqualVector(NoiseAdder()([]), [])

    def testNegatives(self):
        input = [1, -1, 2, -2, 3, -3]
        output = NoiseAdder(level = -10)(input)
        k = db2lin(-10.0)
        self.assert_( all(abs(output - input) <= k))

    def testFixSeed(self):
        a=NoiseAdder(fixSeed=True)(zeros(10))
        b=NoiseAdder(fixSeed=True)(zeros(10))
        self.assertEqualVector(a,b)


suite = allTests(TestNoiseAdder)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

