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

class TestAfterMaxToBeforeMaxEnergyRatio(TestCase):

    def testEmpty(self):
        self.assertComputeFails(AfterMaxToBeforeMaxEnergyRatio(), [])

    def testZero(self):
        self.assertComputeFails(AfterMaxToBeforeMaxEnergyRatio(), [0]*100)

    def testOne(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()([1234]), 1)

    def testAscending(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()(range(10)) < 1, True)

    def testDescending(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()(range(9, -1, -1)) > 1, True)

    def testMaxInclusion(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()([100, 1000, 100]), 1)

    def testOnePitch(self):
        self.assertEqual(AfterMaxToBeforeMaxEnergyRatio()([100]), 1)


suite = allTests(TestAfterMaxToBeforeMaxEnergyRatio)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

