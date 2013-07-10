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
import numpy

class TestFlatnessSfx(TestCase):

    def testEmpty(self):
        self.assertComputeFails(FlatnessSFX(), [])

    def testZero(self):
        self.assertEqual(FlatnessSFX()([0]*100), 1.0)

    def testOne(self):
        self.assertEqual(FlatnessSFX()([1234]), 1.0)

    def testFlat(self):
        self.assertEqual(FlatnessSFX()([0.5]*100), 1.0)

    def testSteep(self):
        self.assertEqual(FlatnessSFX()([0, 1]) > 1.0, True)

    def testRegression(self):
        self.assertAlmostEqual(FlatnessSFX()(list(numpy.linspace(0.0, 1.0, 100))), 4.7500004768371582, 1e-6)


suite = allTests(TestFlatnessSfx)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
