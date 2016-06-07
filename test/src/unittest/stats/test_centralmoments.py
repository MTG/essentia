#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

class TestCentralMoments(TestCase):

    def testZero(self):
        n = 1000
        cm = CentralMoments(mode = 'pdf', range = n-1)
        self.assert_(all(cm(zeros(n)) == 0))

    def testEmptyOrOne(self):
        self.assertComputeFails(CentralMoments(), [])
        self.assertComputeFails(CentralMoments(), [23])

    def testRegression(self):
        inputArray = readVector(join(filedir(), 'stats/input.txt'))
        distShape = readVector(join(filedir(), 'stats/distributionshape.txt'))

        moments = CentralMoments(mode = 'pdf', 
                                 range = len(inputArray)-1)(inputArray)

        self.assertAlmostEqual(moments[2], distShape[5])
        self.assertAlmostEqual(moments[3], distShape[6])
        self.assertAlmostEqual(moments[4], distShape[7])

        # test the 'sample' mode
        moments = CentralMoments(mode='sample')([0., 1, 1, 2, 3])
        distShape = [1.0, 0.0, 1.04, 0.28800000000000037, 2.1152000000000002]

        self.assertAlmostEqual(moments[2], distShape[2])
        self.assertAlmostEqual(moments[3], distShape[3])
        self.assertAlmostEqual(moments[4], distShape[4])


suite = allTests(TestCentralMoments)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
