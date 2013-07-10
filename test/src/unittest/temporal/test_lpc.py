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


class TestLPC(TestCase):

    def testZero(self):
        c, r = LPC(order=10)(zeros(1024))
        self.assertEqualVector(c, zeros(11))
        self.assertEqualVector(r, zeros(10))

    def testRegression(self):
        # got values from before the switch from crossCorrelation to autoCorrelation
        inputData = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 ]
        coeffs = [ 1.          , -1.7860189676,  0.7943785787, -0.0000018278,
                   0.0000157348, -0.0000236177,  0.0000142327,  0.0000024615,
                  -0.0000063489,  0.1693837941, -0.1610202342]
        reflection = [ 0.9850746393, -0.8639101982, -0.0128436843,  0.0014869192,
                       0.0185417943,  0.0385499857,  0.061892774 ,  0.0891031325,
                       0.1213476434,  0.1610202342]

        c, r = LPC(order=10)(inputData)

        # we need to test the absolute value here, not the relative one (well,
        # actually it needs to be relative to the first coeff, which is 1...)
        self.assertAlmostEqualVector(c-array(coeffs), zeros(11), 1e-3)
        self.assertAlmostEqualVector(r-array(reflection), zeros(10), 1e-3)

    def testMatlab(self):
        # same inputData as previous function
        coeffs = [  1.000000000000000,  -1.786020583190265,   0.794382504287881,   0.000000000000216,
                   -0.000000000000067,   0.000000000000015,  -0.000000000000093,   0.000000000000232,
                   -0.000000000000288,   0.169382504288377,  -0.161020583190467 ]
        inputData = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 ]

        c, r = LPC(order=10)(inputData)

        self.assertAlmostEqualVector(c-array(coeffs), zeros(11), 1e-3)

    def testRegression2(self):
        # test found here: http://www.nabble.com/Wrong-results-for-lpc-in-TSA-package-td17273766.html
        impulse = readVector(join(filedir(), 'impulse.txt'))
        coeffs = [ 1.000000000000000,  -1.755371111771445,   0.902500000000000 ]

        c, r = LPC(order=2)(impulse)

        self.assertAlmostEqualVector(c-array(coeffs), zeros(3), 1e-3)

    def testRegression3(self):
        # back to back tests from old svn repository:
        input = readVector(join(filedir(), 'lpc', 'input.txt'))
        expected = readVector(join(filedir(), 'lpc', 'output.txt'))

        c, r = LPC(order=10)(input)
        self.assertAlmostEqualVector(array(c), expected, 5e-6)

    def testWarpedLPC(self):
        # back to back tests from old svn repository:
        input = readVector(join(filedir(), 'warpedlpc', 'input.txt'))
        expected = readVector(join(filedir(), 'warpedlpc', 'output.txt'))

        c, r = LPC(order=4, type='warped')(input)
        self.assertAlmostEqualVector(array(c), expected, 5e-6)

    def testInvalidInput(self):
        # can't have an order > input.size:
        self.assertComputeFails(LPC(order=5),([1,2,3]))

    def testInvalidParam(self):
        self.assertConfigureFails(LPC(), {'order':1}) # order must be [2, inf]
        self.assertConfigureFails(LPC(), {'sampleRate':0})
        self.assertConfigureFails(LPC(), {'type':'unknown'}) # type = {regular,warped}


suite = allTests(TestLPC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
