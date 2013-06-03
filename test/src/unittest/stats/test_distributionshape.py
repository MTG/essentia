#!/usr/bin/env python

from essentia_test import *

class TestDistributionShape(TestCase):

    def testZero(self):
        n = 1000
        cm = CentralMoments(range = n-1)
        ds = DistributionShape()
        self.assertEqualVector(ds(cm(zeros(n))), [0, 0, -3])

    def testInvalidInput(self):
        self.assertComputeFails(DistributionShape(), [])
        self.assertComputeFails(DistributionShape(), ones(1000))

    def testSpecialMoments(self):
        specialMoments = [69, 1999, 0, 21, 18]
        (spread, skewness, kurtosis) = DistributionShape()(specialMoments)

        self.assertEqual(spread, 0.)
        self.assertEqual(skewness, 0.)
        self.assertEqual(kurtosis, -3.0)

    def testRegression(self):
        inputArray = readVector(join(filedir(), 'stats/input.txt'))
        distShape = readVector(join(filedir(), 'stats/distributionshape.txt'))
        
        moments = CentralMoments(range = len(inputArray)-1)(inputArray)
        spread, skewness, kurtosis = DistributionShape()(moments)

        self.assertAlmostEqual(spread, distShape[9], 1e-6)
        self.assertAlmostEqual(skewness, distShape[10], 1e-6)
        self.assertAlmostEqual(kurtosis, distShape[11], 1e-6)

suite = allTests(TestDistributionShape)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

