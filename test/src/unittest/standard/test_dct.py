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


class TestDCT(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(DCT(), { 'inputSize': 0, 'outputSize': 2 })
        self.assertConfigureFails(DCT(), { 'inputSize': 6, 'outputSize': 0 })


    def testRegression(self):
        # values from Matlab/Octave
        inputArray = [ 0, 0, 1, 0, 1 ]
        expected = [ 0.89442719099, -0.60150095500, -0.12078825843, -0.37174803446, 0.82789503961 ]
        self.assertAlmostEqualVector(DCT(outputSize=len(inputArray))(inputArray), expected, 1e-6)


    def testZero(self):
        self.assertEqualVector(DCT(outputSize=10)(zeros(20)), zeros(10))

    def testInvalidInput(self):
        self.assertComputeFails(DCT(), []) # = testEmpty
        self.assertComputeFails(DCT(outputSize = 10), [ 0, 2, 4 ])




suite = allTests(TestDCT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
