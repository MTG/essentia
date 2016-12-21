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


class TestIDCT(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(IDCT(), { 'inputSize': 0, 'outputSize': 2 })
        self.assertConfigureFails(IDCT(), { 'inputSize': 6, 'outputSize': 0 })


    def testRegression(self):
        # values from Matlab/Octave
        inputArray = [ 0.89442718, -0.60150099, -0.12078822, -0.37174806,  0.82789522]
        expected = [ 0, 0, 1, 0, 1 ]
        self.assertAlmostEqualVector(IDCT(outputSize=len(expected), inputSize = len(inputArray))(inputArray), expected, 1e-6)

    def testLifteringRegression(self):
        # DCT III and Liftening computed using PLP and RASTA matlab toolbox. 
        # A big tolerance is necessary due to the smoothing caused by the smaller amount of bins in the DCT domain.
        
        inputArray = [ 1.89736652,  0.95370573,  3.39358997, -3.35009956]
        expected = [1, 1, 0, 0, 1]

        self.assertAlmostEqualVector(IDCT(inputSize=len(inputArray), 
                                         outputSize=len(expected), 
                                         dctType = 3, 
                                         liftering = 22)(inputArray), expected, 1e0)

    def testZero(self):
        self.assertEqualVector(IDCT(outputSize=10)(zeros(5)), zeros(10))

    def testInvalidInput(self):
        self.assertComputeFails(IDCT(), []) # = testEmpty
        self.assertComputeFails(IDCT(outputSize = 2, inputSize = 1), [ 0, 2, 4 ])




suite = allTests(TestIDCT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
