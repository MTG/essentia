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
from essentia.standard import MonoLoader, LoopBpmConfidence

class TestLoopBpmConfidence(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(LoopBpmConfidence(), {'sampleRate': -1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()        

        """
         Confidence figures are obtained from previous runs algorithm as of
         code baseline on 10-Feb-2021
         In keeping with regression tests principles, no future changes of algorithm
         should break these tests.
         Test for different BPMs starting with correct one(125),
         and check the confidence levels obtained from previous runs.
         The 3rd param in assertAlmostEqual() has been optimised to 0.02.
        """        

        bpmEstimate = 125
        expectedConfidence = 1
        confidence = LoopBpmConfidence()(audio, bpmEstimate)        
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02)

        bpmEstimate = 120
        expectedConfidence = 0.87
        confidence = LoopBpmConfidence()(audio, bpmEstimate)        
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02)

        bpmEstimate = 70
        expectedConfidence = 0.68
        confidence = LoopBpmConfidence()(audio, bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02)

        """
         Apply same principle above to Sub-Sections of Audio.       
         The following confidence measurements were taken on 11 Feb. 2021
         for a given set of parameters on different audio subsections.
         A segment of techno_loop gives a non-ideal estimation of BPM.
         
         Subsection [1000:6000], (Randomly chosen points, section 1)
         BPM Estimate: 125: Confidence: 0.5275888442993164
         BPM Estimate: 120: Confidence: 0.5464853048324585
         BPM Estimate: 70: Confidence: 0.7354497313499451
         Subsection [22000:38000], (Randomly chosen points, section 2)
         BPM Estimate: 125: Confidence: 0.5117157697677612
         BPM Estimate: 120: Confidence: 0.4512471556663513
         BPM Estimate: 70: Confidence: 0.15343916416168213
         The ASSERTs will check for these to two places of decimal
        
         These checks act as integrity checks. If any future changes break these asserts,
         then this will indicate that accuracy of algorithm has changed.
         The 3rd param in assertAlmostEqual() has been optimised, lowest possible value chosen.
        """

        # Randomly chosen points, Section 1
        samplePoint1 = 1000
        samplePoint2 = 6000

        bpmEstimate = 125
        expectedConfidence = 0.53
        confidence = LoopBpmConfidence()(audio[samplePoint1:samplePoint2], bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02) 

        bpmEstimate = 120
        expectedConfidence = 0.55
        confidence = LoopBpmConfidence()(audio[samplePoint1:samplePoint2], bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02) 
        
        bpmEstimate = 70
        expectedConfidence = 0.74
        confidence = LoopBpmConfidence()(audio[samplePoint1:samplePoint2], bpmEstimate)            
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02) 

        # Randomly chosen points, Section 2
        samplePoint1 = 22000
        samplePoint2 = 38000
        expectedEstimate = 100
        
        bpmEstimate = 125
        expectedConfidence = 0.51
        confidence = LoopBpmConfidence()(audio[samplePoint1:samplePoint2], bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02) 

        bpmEstimate = 120
        expectedConfidence = 0.45
        confidence = LoopBpmConfidence()(audio[samplePoint1:samplePoint2], bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.02) 
        
        bpmEstimate = 70
        expectedConfidence = 0.15
        confidence = LoopBpmConfidence()(audio[samplePoint1:samplePoint2], bpmEstimate)            
        self.assertAlmostEqual(expectedConfidence, confidence, 0.05) 
        
    def testEmpty(self):
        # Zero estimate check results in zero confidence
        emptyAudio = []
        bpmEstimate = 0
        confidence = LoopBpmConfidence()(emptyAudio, bpmEstimate)                  
        self.assertEquals(0, confidence)

        # Non-zero estimate check results in zero confidence
        emptyAudio = []
        bpmEstimate = 125
        confidence = LoopBpmConfidence()(emptyAudio, bpmEstimate)                  
        self.assertEquals(0, confidence)

    def testZero(self):
        zeroAudio = zeros(1000)
        bpmEstimate = 0
        confidence = LoopBpmConfidence()(zeroAudio, bpmEstimate)
        self.assertEquals(0, confidence)

        # Non-zero estimate check results in non-zero confidence
        # FIX-ME: Maybe given contant zero input, we should expect zero confidence
        bpmEstimate = 125
        confidence = LoopBpmConfidence()(zeroAudio, bpmEstimate)
        self.assertNotEquals(0, confidence)
         
    def testConstantInput(self):
        onesAudio = ones(100)
        bpmEstimate = 0
        confidence = LoopBpmConfidence()(onesAudio, bpmEstimate)
        self.assertEquals(0, confidence)

        # Non-zero estimate check results in non-zero confidence
        # FIX-ME: Maybe given contant input, we should expect zero confidence
        bpmEstimate = 125
        confidence = LoopBpmConfidence()(onesAudio, bpmEstimate)
        self.assertNotEquals(0, confidence)
         
suite = allTests(TestLoopBpmConfidence)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
