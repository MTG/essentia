#!/usr/bin/env python

from essentia_test import *

class TestRawMoments(TestCase):

    def testZero(self):
        n = 1000
        rawMoments = RawMoments(range = n-1)
        self.assert_(all(rawMoments(zeros(n)) == 0))

    def testEmptyOrOne(self):
        self.assertComputeFails(RawMoments(), [])
        self.assertComputeFails(RawMoments(), [23])

    def testRegression(self):
        input = readVector(join(filedir(), 'stats/input.txt'))
        range = len(input)-1
        total = sum(input)

        expectedMoments = [0]*5

        expectedMoments[0] = 1
        expectedMoments[1] = sum( [pow(freq,1)*input[freq] for freq in xrange(len(input))] ) / total
        expectedMoments[2] = sum( [pow(freq,2)*input[freq] for freq in xrange(len(input))] ) / total
        expectedMoments[3] = sum( [pow(freq,3)*input[freq] for freq in xrange(len(input))] ) / total
        expectedMoments[4] = sum( [pow(freq,4)*input[freq] for freq in xrange(len(input))] ) / total


        moments = RawMoments(range = range)(input)

        self.assertAlmostEqual(moments[0], expectedMoments[0])
        self.assertAlmostEqual(moments[1], expectedMoments[1], 1e-6)
        self.assertAlmostEqual(moments[2], expectedMoments[2], 1e-6)
        self.assertAlmostEqual(moments[3], expectedMoments[3], 1e-6)
        self.assertAlmostEqual(moments[4], expectedMoments[4], 1e-6)


suite = allTests(TestRawMoments)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
