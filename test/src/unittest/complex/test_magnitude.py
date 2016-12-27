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
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/


from essentia_test import *
from math import *

class TestMagnitude(TestCase):

    def testZero(self):
        inputc = numpy.array([ complex() ] * 4, dtype='c8')

        self.assertEqualVector(Magnitude()(inputc), zeros(4))

    def testEmpty(self):
        self.assertEqualVector(Magnitude()(numpy.array([],dtype='c8')), [])

    def testRegression(self):
        inputc = [ (1, -5), (2, -6), (-3, 7), (-4, 8) ]
        inputc = numpy.array([ complex(*c) for c in inputc ], dtype='c8')
        expected = array([ abs(c) for c in inputc ])

        self.assertAlmostEqualVector(Magnitude()(inputc), expected)


suite = allTests(TestMagnitude)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

