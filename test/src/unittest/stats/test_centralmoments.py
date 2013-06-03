#!/usr/bin/env python

from essentia_test import *

class TestCentralMoments(TestCase):

    def testZero(self):
        n = 1000
        cm = CentralMoments(range = n-1)
        self.assert_(all(cm(zeros(n)) == 0))

    def testEmptyOrOne(self):
        self.assertComputeFails(CentralMoments(), [])
        self.assertComputeFails(CentralMoments(), [23])

    def testRegression(self):
        inputArray = readVector(join(filedir(), 'stats/input.txt'))
        distShape = readVector(join(filedir(), 'stats/distributionshape.txt'))
        
        moments = CentralMoments(range = len(inputArray)-1)(inputArray)

        self.assertAlmostEqual(moments[2], distShape[5])
        self.assertAlmostEqual(moments[3], distShape[6])
        self.assertAlmostEqual(moments[4], distShape[7])


suite = allTests(TestCentralMoments)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
