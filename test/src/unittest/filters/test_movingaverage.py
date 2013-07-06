#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/



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
