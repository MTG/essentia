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
from essentia.standard import MonoLoader, LoopBpmEstimator
import numpy as np

class TestLoopBpmEstimator(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(LoopBpmEstimator(), {'confidenceThreshold': -1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()        
      
        expectedEstimate= 125
        estimate = LoopBpmEstimator(confidenceThreshold=0.95)(audio)            
        self.assertEqual(expectedEstimate, estimate) 
        
        # Test for upperbound of Confidence Threshold
        estimate = LoopBpmEstimator(confidenceThreshold=1)(audio)     
        self.assertEqual(expectedEstimate, estimate)

        # Test for lowerbound of Confidence Threshold
        estimate = LoopBpmEstimator(confidenceThreshold=0)(audio)   
        self.assertEqual(expectedEstimate, estimate)

        # Further integrity checks at randomly chosen points in audio.
        # If any future changes break these asserts,
        # then this will indicates algorithm is compromised.

        # Randomly chosen points, Section 1
        # A segment of techno_loop gives a non-ideal estimation of BPM.
        samplePoint1 = 1000
        samplePoint2 = 6000
        expectedEstimate = 100.0

        # 0% confidence expectation triggers non-zero bpm close to estimate result in this section 1
        estimate = LoopBpmEstimator(confidenceThreshold=0.0)(audio[samplePoint1:samplePoint2])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # 10% confidence expectation triggers non-zero result in this section 1
        estimate = LoopBpmEstimator(confidenceThreshold=0.1)(audio[samplePoint1:samplePoint2])                
        self.assertNotEqual(0, estimate) 

        # Higher than 20% confidence expectation triggers non-zero bpm result in this section 1
        estimate = LoopBpmEstimator(confidenceThreshold=0.25)(audio[samplePoint1:samplePoint2])                
        self.assertNotEqual(0, estimate) 

        # Higher than 50% confidence expectation triggers non-zero bpm result in this section 1
        estimate = LoopBpmEstimator(confidenceThreshold=0.55)(audio[samplePoint1:samplePoint2])                
        self.assertAlmostEqual(expectedEstimate, estimate, 5) 

        # High confidence expectation triggers 0 bpm result in this section 1
        estimate = LoopBpmEstimator(confidenceThreshold=0.95)(audio[samplePoint1:samplePoint2])            
        self.assertEqual(0, estimate) 

        # Randomly chosen points, Section 2
        # A segment of techno_loop gives a non-ideal estimation of BPM.        
        samplePoint1 = 22000
        samplePoint2 = 38000
        expectedEstimate = 100.0

        # 0% confidence expectation triggers non-zero bpm close to estimate result in this section 2
        estimate = LoopBpmEstimator(confidenceThreshold=0.0)(audio[samplePoint1:samplePoint2])             
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # 10% confidence expectation triggers non-zero bpm result in this section 2
        estimate = LoopBpmEstimator(confidenceThreshold=0.1)(audio[samplePoint1:samplePoint2])                
        self.assertNotEqual(0, estimate) 

        # Higher than 20% confidence expectation triggers 0 bpm result in this section 2
        estimate = LoopBpmEstimator(confidenceThreshold=0.25)(audio[samplePoint1:samplePoint2])                
        self.assertEqual(0, estimate) 

        # Higher than 50% confidence expectation triggers 0 bpm result in this section 2
        estimate = LoopBpmEstimator(confidenceThreshold=0.55)(audio[samplePoint1:samplePoint2])                
        self.assertEqual(0, estimate) 

        # High confidence expectation triggers 0 bpm result in this section 2
        estimate = LoopBpmEstimator(confidenceThreshold=0.95)(audio[samplePoint1:samplePoint2])            
        self.assertEqual(0, estimate) 

    # A series of assert checks on the BPM estimator for empty, zero or constant signals.
    # LoopBpmEstimator uses the percival estimator internally.
    # The runtime errors have their origin in that algorithm.
    def testEmpty(self):
        emptyAudio = []
        self.assertRaises(RuntimeError, lambda: LoopBpmEstimator()(emptyAudio))

    def testZero(self):
        zeroAudio = zeros(1000)
        self.assertRaises(RuntimeError, lambda: LoopBpmEstimator()(zeroAudio))

    def testConstantInput(self):
        onesAudio = ones(100)
        self.assertRaises(RuntimeError, lambda: LoopBpmEstimator()(onesAudio))


suite = allTests(TestLoopBpmEstimator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
