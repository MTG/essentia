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
         In keeping with regression tests principles, no future changes of the algorithm
         should break these tests.
         Test for different BPMs starting with correct one: 125,
         and check the confidence levels obtained from previous runs.
         The 3rd param in assertAlmostEqual() has been optimised to 0.01.
         We chose this reasonable margin since the expected confidence benchmark 
         has been rounded to two places of decimal.
        """        

        bpmEstimate = 125
        expectedConfidence = 1
        confidence = LoopBpmConfidence()(audio, bpmEstimate)        
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01)

        bpmEstimate = 120
        expectedConfidence = 0.87
        confidence = LoopBpmConfidence()(audio, bpmEstimate)        
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01)

        bpmEstimate = 70
        expectedConfidence = 0.68
        confidence = LoopBpmConfidence()(audio, bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01)

        """
         Apply same principle above but now to Sub-Sections of Audio,
         truncating a bit at the end
         The following confidence measurements were taken on 11 Feb. 2021
         for a given set of parameters on different audio subsections.
         A segment of techno_loop gives a non-ideal estimation of BPM
         so we expect the confidence to drop.
       
         Subsection [0:len90]
         BPM Estimate: 125: Confidence: 0.19992440938949585
         BPM Estimate: 120: Confidence: 0.4305669069290161
         BPM Estimate: 70: Confidence: 0.5011640191078186

         Subsection [len01:len90],
         BPM Estimate: 125: Confidence: 0.9199735522270203
         BPM Estimate: 120: Confidence: 0.3631746172904968
         BPM Estimate: 70: Confidence: 0.7951852083206177
         The ASSERTs will check for these to two places of decimal
        
         These checks act as integrity checks. If any future changes break these asserts,
         then this will indicate that accuracy of algorithm has changed.
         The 3rd param in assertAlmostEqual() has been optimised, lowest possible value chosen.
        """

        # Define Markers for significant meaningful subsections 
        # consistent with sampling rate
        len90 = int(0.9*len(audio))  # End point for 90% loop and subsection        
        len01 = int(0.01*len(audio)) # Begin point for subsection loop 

        # Truncated Audios 90%
        bpmEstimate = 125
        expectedConfidence = 0.2 # rounded from measured 0.19992440938949585
        confidence = LoopBpmConfidence()(audio[0:len90], bpmEstimate)              
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01) 

        bpmEstimate = 120
        expectedConfidence = 0.43 # rounded from measured 0.4305669069290161
        confidence = LoopBpmConfidence()(audio[0:len90], bpmEstimate)                          
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01) 
        
        bpmEstimate = 70
        expectedConfidence = 0.50 # rounded from measured 0.5011640191078186
        confidence = LoopBpmConfidence()(audio[0:len90], bpmEstimate)                     
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01)
        
        # Truncated Audios 90% and 1% at beginning
        bpmEstimate = 125
        expectedConfidence = 0.92 # rounded from measured 0.9199735522270203
        confidence = LoopBpmConfidence()(audio[len01:len90], bpmEstimate)                         
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01) 

        bpmEstimate = 120
        expectedConfidence =  0.36 # rounded from measured 0.3631746172904968
        confidence = LoopBpmConfidence()(audio[len01:len90], bpmEstimate)           
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01) 
        
        bpmEstimate = 70 
        expectedConfidence =  0.79 # rounded from measured 0.7951852083206177
        confidence = LoopBpmConfidence()(audio[len01:len90], bpmEstimate)
        self.assertAlmostEqual(expectedConfidence, confidence, 0.01) 
        
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
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))() 
        zeroAudio = zeros(len(audio))
        bpmEstimate = 0
        confidence = LoopBpmConfidence()(zeroAudio, bpmEstimate)
        self.assertEquals(0, confidence)

        # Non-zero estimate check results in non-zero confidence
        # FIXME: Maybe given contant zero input, we should expect zero confidence
        bpmEstimate = 125
        confidence = LoopBpmConfidence()(zeroAudio, bpmEstimate)
        self.assertNotEquals(0, confidence)
         
    def testConstantInput(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))() 
        onesAudio = ones(len(audio))
        bpmEstimate = 0
        confidence = LoopBpmConfidence()(onesAudio, bpmEstimate)
        self.assertEquals(0, confidence)

        # Non-zero estimate check results in non-zero confidence
        # FIXME: Maybe given contant input, we should expect zero confidence
        bpmEstimate = 125
        confidence = LoopBpmConfidence()(onesAudio, bpmEstimate)
        self.assertNotEquals(0, confidence)
         
suite = allTests(TestLoopBpmConfidence)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
