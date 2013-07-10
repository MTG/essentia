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

class TestUnaryOperator(TestCase):

    testInput = [1,2,3,4,3.4,-5.0008, 100034]

    def testEmpty(self):
        self.assertEqualVector(UnaryOperator()([]), [])

    def testOne(self):
        self.assertEqualVector(UnaryOperator(type="identity")([101]), [101])

    def testAbs(self):
        self.assertAlmostEqualVector(UnaryOperator(type="abs")(self.testInput),
            [1,2,3,4,3.4,5.0008,100034])

    def testLog10(self):
        self.assertAlmostEqualVector(
            UnaryOperator(type="log10")(self.testInput),
            [0., 0.30103001, 0.4771212637, 0.60206002, 0.5314789414, -30., 5.0001478195])

    def testLog(self):
        self.assertAlmostEqualVector(
            UnaryOperator(type="log")(self.testInput),
            [0., 0.6931471825, 1.0986123085, 1.3862943649, 1.223775506, -69.0775527954, 11.5132656097])

    def testLn(self):
        self.assertAlmostEqualVector(UnaryOperator(type="ln")(self.testInput),
            [0, 0.693147181, 1.098612289, 1.386294361, 1.223775432, -69.07755279, 11.513265407])

    def testLin2Db(self):
        self.assertAlmostEqualVector(
            UnaryOperator(type="lin2db")(self.testInput),
            [0., 3.01029992, 4.77121258, 6.02059984, 5.3147893, -90., 50.00147629])

    def testDb2Lin(self):
        # remove the last element because it causes an overflow because it is
        # too large
        self.assertAlmostEqualVector(
                UnaryOperator(type="db2lin")(self.testInput[:-1]),
                [1.25892544, 1.58489323, 1.99526227, 2.51188636, 2.18776178, 0.3161695],
                2e-7)

    def testSine(self):
        self.assertAlmostEqualVector(UnaryOperator(type="sin")(self.testInput),
            [0.841470985, 0.909297427, 0.141120008, -0.756802495, -0.255541102, 0.958697038, -0.559079868], 1e-6)

    def testCosine(self):
        self.assertAlmostEqualVector(UnaryOperator(type="cos")(self.testInput),
            [0.540302306, -0.416146837, -0.989992497, -0.653643621, -0.966798193, 0.284429234, 0.829113805], 1e-6)

    def testSqrt(self):
        # first take abs so we won't take sqrt of a negative (that test comes later)
        absInput = UnaryOperator(type="abs")(self.testInput)
        self.assertAlmostEqualVector(UnaryOperator(type="sqrt")(absInput),
            [1, 1.414213562, 1.732050808, 2, 1.843908891, 2.236246856, 316.281520168])

    def testSqrtNegative(self):
        self.assertComputeFails(UnaryOperator(type="sqrt"),([0, -1, 1]))

    def testSquare(self):
        self.assertAlmostEqualVector(UnaryOperator(type="square")(self.testInput),
            [1, 4, 9, 16, 11.56, 25.0080006, 10006801156])

    def testInvalidParam(self):
        self.assertConfigureFails(UnaryOperator(), {'type':'exp'})


suite = allTests(TestUnaryOperator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
