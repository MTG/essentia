#!/usr/bin/env python

from essentia_test import *

class TestCentroid(TestCase):

    def testZero(self):
        self.assertEqual(Centroid()(zeros(1000)), 0)

    def testEmptyOrOne(self):
        self.assertComputeFails(Centroid(), [])
        self.assertComputeFails(Centroid(), [23])


    def testRegression(self):
        inputArray = readVector(join(filedir(), 'stats/input.txt'))
        distShape = readVector(join(filedir(), 'stats/distributionshape.txt'))

        centroid = Centroid(range = len(inputArray)-1)(inputArray)

        self.assertAlmostEqual(centroid, distShape[8], 1e-6)



suite = allTests(TestCentroid)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

