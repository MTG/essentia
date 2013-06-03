#!/usr/bin/env python

from essentia_test import *

class TestPowerMean(TestCase):

    def testEmpty(self):
        self.assertComputeFails(PowerMean(), [])
        self.assertComputeFails(PowerMean(power=0), [])

    def testZero(self):
        zeroInput = [0]*10
        self.assertEquals(PowerMean()(zeroInput), 0)

        # this test passes, but its behavior is undefined
        #self.assertAlmostEqual(PowerMean(power=0), 0);

    def testOne(self):
        oneInput = [100]
        self.assertAlmostEqual(PowerMean()(oneInput), oneInput[0])
        self.assertAlmostEqual(PowerMean(power=0)(oneInput), oneInput[0])
        self.assertAlmostEqual(PowerMean(power=-2)(oneInput), oneInput[0])
        self.assertAlmostEqual(PowerMean(power=6)(oneInput), oneInput[0], 1e-6)

    def testMulti(self):
        input = [5, 8, 4, 9, 1]
        self.assertAlmostEqual(PowerMean()(input), 5.4)
        self.assertAlmostEqual(PowerMean(power=0)(input), 4.28225474)
        self.assertAlmostEqual(PowerMean(power=-3)(input), 1.69488507)
        self.assertAlmostEqual(PowerMean(power=4)(input), 6.93105815)

    def testNegatives(self):
        input = [3, 7, -45, 2, -1, 0]
        self.assertComputeFails(PowerMean(), input)
        self.assertComputeFails(PowerMean(power=0), input)

    def testRational(self):
        input = [3.1459, 0.4444, .00002]
        self.assertAlmostEqual(PowerMean()(input), 1.19677333)
        self.assertAlmostEqual(PowerMean(power=0)(input), 0.0303516976)
        self.assertAlmostEqual(PowerMean(power=-5.1)(input), 2.48075104e-5)
        self.assertAlmostEqual(PowerMean(power=2.3)(input), 1.96057772)


suite = allTests(TestPowerMean)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
