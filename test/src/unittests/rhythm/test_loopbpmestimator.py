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


from essentia_test import *
from essentia.standard import MonoLoader, LoopBpmEstimator
import numpy as np

class TestLoopBpmEstimator(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(LoopBpmEstimator(), {'confidenceThreshold': -1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()        
      
        expectedEstimate = 125
        estimate = LoopBpmEstimator(confidenceThreshold=0.95)(audio)            
        self.assertEqual(expectedEstimate, estimate) 
        
        # Test for upperbound of Confidence Threshold
        estimate = LoopBpmEstimator(confidenceThreshold=1)(audio)     
        self.assertEqual(expectedEstimate, estimate)

        # Test for lowerbound of Confidence Threshold
        estimate = LoopBpmEstimator(confidenceThreshold=0)(audio)   
        self.assertEqual(expectedEstimate, estimate)

        # Define Markers for significant meaningful subsections 
        # consistent with sampling rate
        len90 = int(0.9*len(audio))  # End point for 90% loop
        len50 = int(0.5*len(audio))  # Mid point                
        len01 = int(0.01*len(audio)) # Begin point for subsection loop 

        # If any future changes break these asserts,
        # then this will indicate algorithm is compromised
        # First test with default confidence value (0.95)
        # Margin of 0.1 chosen based on accuracy of observed BPMs.

        # Loop is broken at end, we expect zero output
        estimate = LoopBpmEstimator()(audio[0:len90])                
        self.assertEqual(0, estimate)         
      
        # Loop is broken at beginning and end, we expect zero output.
        estimate = LoopBpmEstimator()(audio[len01:len90])                
        self.assertEqual(0, estimate) 
        
        # When the loop is halved we get close to the expected estimate to 1 place of decimal.
        estimate = LoopBpmEstimator()(audio[0:len50])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # Repeat above tests with non-default confidence value of 0.5.
        # Loop is broken at end, we expect zero output        
        estimate = LoopBpmEstimator(confidenceThreshold=0.5)(audio[0:len90])                
        self.assertEqual(0, estimate)         
        
        # Lower confidence, results in value close to expected estimate.
        estimate = LoopBpmEstimator(confidenceThreshold=0.5)(audio[len01:len90])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 
      
        # Lower confidence, results in value close to expected estimate.
        estimate = LoopBpmEstimator(confidenceThreshold=0.5)(audio[0:len50])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # Repeat above tests with non-default confidence value of 0.2.
        # Loop is broken at end, we expect zero output
        estimate = LoopBpmEstimator(confidenceThreshold=0.2)(audio[0:len90])                
        self.assertEqual(0, estimate)         
        
        # Lower confidence, results in value close to expected estimate.
        estimate = LoopBpmEstimator(confidenceThreshold=0.2)(audio[len01:len90])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1)

        # Lower confidence, results in value close to expected estimate.
        estimate = LoopBpmEstimator(confidenceThreshold=0.2)(audio[0:len50])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

    # A series of assert checks on the BPM estimator for empty, zero or constant signals.
    # LoopBpmEstimator uses the percival estimator internally.
    # The runtime errors have their origin in that algorithm.
    def testEmpty(self):
        emptyAudio = []
        self.assertRaises(RuntimeError, lambda: LoopBpmEstimator()(emptyAudio))

    def testZero(self):        
        zeroAudio = zeros(100000)
        self.assertRaises(RuntimeError, lambda: LoopBpmEstimator()(zeroAudio))

    def testConstantInput(self):
        onesAudio = ones(100000)
        estimate = LoopBpmEstimator(confidenceThreshold=0.95)(onesAudio)            
        self.assertEqual(0, estimate)

suite = allTests(TestLoopBpmEstimator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
