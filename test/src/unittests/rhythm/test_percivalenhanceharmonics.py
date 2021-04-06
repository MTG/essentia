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


from numpy import *
from essentia_test import *
from essentia.standard import MonoLoader, PercivalEnhanceHarmonics

class TestPercivalEnhanceHarmonics(TestCase):
             
    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()        

        calculatedEnhancedHarmonics = PercivalEnhanceHarmonics()(audio)

        """
        This code stores reference values in a file for later loading.
        save('enhancedharmonics.npy', calculatedEnhancedHarmonics)
        """        
       
        # Reference samples are loaded as expected values
        expectedEnhancedHarmonics = load(join(filedir(), 'percival/enhancedharmonics.npy'))
        expectedEnhancedHarmonicsList = expectedEnhancedHarmonics.tolist()

        self.assertAlmostEqualVectorFixedPrecision(calculatedEnhancedHarmonics, expectedEnhancedHarmonicsList,2)


    # A series of assert checks on the BPM estimator for empty, zero or constant signals.
    # PercivalEnhanceHarmonics uses the percival estimator internally.
    # The runtime errors have their origin in that algorithm.
    def testEmpty(self):
        emptyAudio = []
        enhancedHarmonics = PercivalEnhanceHarmonics()(emptyAudio)
        self.assertEqualVector(enhancedHarmonics, [])


    def testZero(self):
        zeroAudio = zeros(100000)
        enhancedHarmonics = PercivalEnhanceHarmonics()(zeroAudio)
        self.assertEqualVector(enhancedHarmonics, zeros(100000))

    def testConstantInput(self):
        onesAudio = ones(100000)
        constantHarmonics = PercivalEnhanceHarmonics()(onesAudio)     
        """
        This code stores reference values in a file for later loading.
        save('constantharmonics.npy', constantHarmonics)     
        """

        # Reference samples are loaded as expected values
        expectedConstantHarmonics = load(join(filedir(), 'percival/constantharmonics.npy'))
        expectedConstantHarmonicsList = expectedConstantHarmonics.tolist()
        self.assertEqualVector(constantHarmonics, expectedConstantHarmonicsList)


    def testResetMethod(self):
        percivalenhanceharmonics = PercivalEnhanceHarmonics()

        self.testRegression()
        percivalenhanceharmonics.reset()
        self.testRegression()


suite = allTests(TestPercivalEnhanceHarmonics)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
