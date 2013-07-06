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


class TestCrossCorrelation(TestCase):

    def testRegression(self):
        testdir = join(filedir(), 'crosscorrelation')
        input1 = readVector(join(testdir, 'input1.txt'))
        input2 = readVector(join(testdir, 'input2.txt'))
        expected = readVector(join(testdir, 'output.txt'))
        xcor = CrossCorrelation(minLag=-5,maxLag=5)
        self.assertAlmostEqualVector(xcor(input1, input2), expected, 1e-5)

    def testEmptyInput(self):
        x = []
        y = []

        self.assertComputeFails(CrossCorrelation(), x, y)

    def testInvalidMinMaxWRTInputs(self):
        x = [1]
        y = [10]

        self.assertAlmostEqualVector(CrossCorrelation(minLag=-1, maxLag=2)(x, y),
                                     [0, 10, 0, 0])

    def testDiffSizeInputs(self):
        x = [1,2,3,4,5]
        y = [-3.546,-65,0,32]

        self.assertAlmostEqualVector(CrossCorrelation(minLag=-3, maxLag=4)(x, y),
                                     [32, 64, 31, -5.546, -42.092, -270.63800, -339.18400, -17.73], 1e-6)

    def testSameSizeInputs(self):
        x = [4.1, 8.6]
        y = [-3.1, 0.4321]

        self.assertAlmostEqualVector(CrossCorrelation(minLag=-1, maxLag=1)(x, y),
                                     [1.77161, -8.99394, -26.66], 1e-6)

    def testMaxSmallerThanMin(self):
        self.assertConfigureFails(CrossCorrelation(), {'minLag': 2, 'maxLag': 1})

    def testMinMaxEqual(self):
        x = [10]
        y = [10]
        self.assertAlmostEqualVector(CrossCorrelation(minLag=0, maxLag=0)(x, y), [100])

    def testZeroSignals(self):
        x = [0]*50
        y = [0]*50
        self.assertAlmostEqualVector(CrossCorrelation(minLag=-49, maxLag=49)(x, y), [0]*(50*2 - 1))

        x = [10]*50
        y = [0]
        self.assertAlmostEqualVector(CrossCorrelation(maxLag=49)(x, y), [0]*50)

    def testEquivalentInputs(self):
        x = [45, -12, 4.783, -0.342, 1]
        self.assertAlmostEqualVector(CrossCorrelation(minLag=-2, maxLag=2)(x, x),
                                     [224.122, -599.373786, 2192.99405, -599.373786, 224.122])


    def testZero(self):
        x = [0]*1024
        y = [0]*100

        result = CrossCorrelation(minLag=-5, maxLag=5)(x, y)

        self.assertAlmostEqualVector(result, [0]*11)


suite = allTests(TestCrossCorrelation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
