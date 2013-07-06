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

class TestPowerMean(TestCase):

    def testEmpty(self):
        self.assertComputeFails(PowerMean(), [])
        self.assertComputeFails(PowerMean(power=0), [])

    def testZero(self):
        zeroInput = [0]*10
        self.assertEquals(PowerMean()(zeroInput), 0)

        # this test passes, but its behavior is undefined
        #self.assertAlmostEqual(PowerMean(power=0), 0);

    def testOne(self):
        oneInput = [100]
        self.assertAlmostEqual(PowerMean()(oneInput), oneInput[0])
        self.assertAlmostEqual(PowerMean(power=0)(oneInput), oneInput[0])
        self.assertAlmostEqual(PowerMean(power=-2)(oneInput), oneInput[0])
        self.assertAlmostEqual(PowerMean(power=6)(oneInput), oneInput[0], 1e-6)

    def testMulti(self):
        input = [5, 8, 4, 9, 1]
        self.assertAlmostEqual(PowerMean()(input), 5.4)
        self.assertAlmostEqual(PowerMean(power=0)(input), 4.28225474)
        self.assertAlmostEqual(PowerMean(power=-3)(input), 1.69488507)
        self.assertAlmostEqual(PowerMean(power=4)(input), 6.93105815)

    def testNegatives(self):
        input = [3, 7, -45, 2, -1, 0]
        self.assertComputeFails(PowerMean(), input)
        self.assertComputeFails(PowerMean(power=0), input)

    def testRational(self):
        input = [3.1459, 0.4444, .00002]
        self.assertAlmostEqual(PowerMean()(input), 1.19677333)
        self.assertAlmostEqual(PowerMean(power=0)(input), 0.0303516976)
        self.assertAlmostEqual(PowerMean(power=-5.1)(input), 2.48075104e-5)
        self.assertAlmostEqual(PowerMean(power=2.3)(input), 1.96057772)


suite = allTests(TestPowerMean)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
