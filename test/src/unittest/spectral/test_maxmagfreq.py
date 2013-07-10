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

class TestMaxMagFreq(TestCase):

    def testEmpty(self):
        self.assertComputeFails(MaxMagFreq(), [])

    def testOne(self):
        self.assertComputeFails(MaxMagFreq(), [1])

    def testZeroFreq(self):
        self.assertAlmostEqual(
                MaxMagFreq()([10, 1, 2, 3]),
                0)

    def testRegression(self):
        self.assertAlmostEqual(
                MaxMagFreq()([3.55,-4.11,0.443,3.9555,2]),
                3 * 22050 / 4.)

    def testNonDefaultSampleRate(self):
        self.assertAlmostEqual(
                MaxMagFreq(sampleRate=10)([1,2,3,4,5]),
                4 * 5 / 4.)

    def testInvalidParam(self):
        self.assertConfigureFails(MaxMagFreq(), {'sampleRate' : 0})

suite = allTests(TestMaxMagFreq)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
