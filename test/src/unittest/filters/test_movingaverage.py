#!/usr/bin/env python

from essentia_test import *


class TestMovingAverage(TestCase):


    def testRegression(self):
        # check moving average for size = 6 and input signal of 10 elements

        input = [1]*10
        expected = [ 1./6, 2./6, 3./6, 4./6., 5./6., 1., 1., 1., 1., 1. ]

        self.assertAlmostEqualVector(MovingAverage(size=6)(input), expected)

    def testOneByOne(self):
        # we compare here that filtering an array all at once or the samples
        # one by one will yield the same result

        input = [1]*10
        expected = [ 1./4, 2./4, 3./4, 1., 1., 1., 1., 1., 1., 1. ]
        filt = MovingAverage(size=4)

        self.assertAlmostEqualVector(filt(input), expected)

        # need to reset the filter here!!
        filt.reset()

        result = []
        for sample in input:
            result += list(filt([sample]))
      
        self.assertAlmostEqualVector(result, expected)
        
        
    def testZero(self):
        self.assertEqualVector(MovingAverage()(zeros(20)), zeros(20))

    def testInvalidParam(self):
        self.assertConfigureFails(MovingAverage(), {'size': 0})


    def testEmpty(self):
        self.assertEqualVector(MovingAverage()([]), [])



suite = allTests(TestMovingAverage)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
