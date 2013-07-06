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

class TestRMS(TestCase):

    def testEmpty(self):
        self.assertComputeFails(RMS(), [])

    def testZero(self):
        result = RMS()([0]*10)
        self.assertAlmostEqual(result, 0)

    def testRegression(self):
        result = RMS()([3, 7.32, -45, 2, -1.453, 0])
        self.assertAlmostEqual(result, 18.680174914420192)

    def testSine(self):
        from numpy import sin, sqrt, pi
        size = 1000
        sine = [sin(440.0*2.0*pi*i/size) for i in range(size)]
        self.assertAlmostEqual(RMS()(sine), 1.0/sqrt(2.0), 1e-6)


suite = allTests(TestRMS)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
