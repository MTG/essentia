#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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


from numpy import *
from essentia_test import *
from essentia.standard import MonoLoader, PercivalEnhanceHarmonics
from math import sin, pi

testdir = join(filedir(), 'percival')

class TestPercivalEnhanceHarmonics(TestCase):
             
    def testRegression(self):
        # Use synthetic audio for Regression Test. This keeps NPY files size low.     
        sr = 44100
        size = 1*sr
        sine1 = [sin(2.0*pi*100.0*i/sr) for i in range(size)]
        sine2 = [sin(2.0*pi*1000.0*i/sr) for i in range(size)]
        fc1 = FrameCutter()
        fc2 = FrameCutter()
        frame1 = fc1(sine1)
        frame2 = fc2(sine2)
        signal = frame1+frame2
        output = AutoCorrelation()(signal)
        calculatedEnhancedHarmonics = PercivalEnhanceHarmonics()(output)

        #This code stores reference values in a file for later loading.
        save('enhancedharmonics.npy', calculatedEnhancedHarmonics)
        # Reference samples are loaded as expected values
        expectedEnhancedHarmonics = load(join(filedir(), 'percival/enhancedharmonics.npy'))
        expectedEnhancedHarmonicsList = expectedEnhancedHarmonics.tolist()
        self.assertAlmostEqualVectorFixedPrecision(calculatedEnhancedHarmonics, expectedEnhancedHarmonicsList,2)


    # A series of assert checks on the BPM estimator for empty, zero or constant signals.
    # PercivalEnhanceHarmonics uses the percival estimator internally.
    # The runtime errors have their origin in that algorithm.
    def testEmpty(self):
        # Define input vector, i.e. the autocorrelation vector
        emptyInput = []
        enhancedHarmonics = PercivalEnhanceHarmonics()(emptyInput)
        self.assertEqualVector(enhancedHarmonics, [])

    def testZero(self):
        # Define input vector, i.e. the autocorrelation vector        
        zeroInput = zeros(1024)
        enhancedHarmonics = PercivalEnhanceHarmonics()(zeroInput)
        self.assertEqualVector(enhancedHarmonics, zeroInput)

    def testConstantInput(self):
        # Define input vector, i.e. the autocorrelation vector        
        onesInput= ones(1024)
        constantHarmonics = PercivalEnhanceHarmonics()(onesInput)     
        #This code stores reference values in a file for later loading.
        save('constantharmonics.npy', constantHarmonics)     

        # Reference samples are loaded as expected values
        expectedConstantHarmonics = load(join(filedir(), 'percival/constantharmonics.npy'))
        expectedConstantHarmonicsList = expectedConstantHarmonics.tolist()
        self.assertEqualVector(constantHarmonics, expectedConstantHarmonicsList)

suite = allTests(TestPercivalEnhanceHarmonics)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
