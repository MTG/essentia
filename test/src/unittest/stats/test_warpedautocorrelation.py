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
testdir = join(filedir(), 'warpedautocorrelation')

class TestWarpedAutoCorrelation(TestCase):

    def testMaxLagLargerThanInputSize(self):
        self.assertComputeFails(
                WarpedAutoCorrelation(maxLag=101),
                [1]*100)

    def testLambdaTooLarge(self):
        # not sure how to set up this test, check the exception that can be
        # thrown in the configure method --rtoscano
        #badSampleRate = ????
        #self.assertConfigureFails(
        #        WarpedAutoCorrelation(),
        #        {'sampleRate': badSampleRate})
        pass

    def testRegression(self):
        input = readVector(join(testdir, 'regression-input.txt'))
        expected = readVector(join(testdir, 'regression-output.txt'))

        output = WarpedAutoCorrelation(maxLag=len(input)-1)(input)

        self.assertAlmostEqualVector(expected, output, 1e-4)

    def testEmpty(self):
        self.assertComputeFails(
                WarpedAutoCorrelation(maxLag=1),
                [])

    def testOne(self):
        self.assertComputeFails(
                WarpedAutoCorrelation(maxLag=1),
                [1])

    def testTwo(self):
        self.assertEqualVector(
                WarpedAutoCorrelation(sampleRate=1000, maxLag=1)([1]*2),
                [2])

    def testZero(self):
        input = [0]*1024
        expected = [0]*10

        result = WarpedAutoCorrelation(maxLag=10)(input)
        self.assertEqualVector(result, expected)


suite = allTests(TestWarpedAutoCorrelation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
