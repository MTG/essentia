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
