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
        self.assertConfigureFails(LoopBpmConfidence(), { 'sampleRate': -1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()        

        # Test for different BPMs starting with correct one then with decreasing accuracy
        # and corresponding levels decreasing confidence.
        # Figures obtained from previous runs algorithm as of baseline 10-Feb-2021
        # In keeping with regression tests principles , no future changes of algorothm
        # should break these tests.
        bpmEstimate = 125
        expectedConfidence = 1
        confidence = LoopBpmConfidence()(audio,bpmEstimate)        
        self.assertAlmostEqual(expectedConfidence, confidence, 0.1)

        bpmEstimate = 120
        expectedConfidence = 0.87
        confidence = LoopBpmConfidence()(audio, bpmEstimate)        
        self.assertAlmostEqual(expectedConfidence, confidence, 0.1)

        bpmEstimate = 70
        expectedConfidence = 0.68
        confidence = LoopBpmConfidence()(audio, bpmEstimate)                
        self.assertAlmostEqual(expectedConfidence, confidence, 0.1)

    def testEmpty(self):
        emptyAudio = []
        bpmEstimate = 0
        expectedConfidence = 0
        confidence = LoopBpmConfidence()(emptyAudio, bpmEstimate)                  
        self.assertEquals(expectedConfidence, confidence)

    def testZero(self):
        zeroAudio = zeros(1000)
        bpmEstimate = 0
        expectedConfidence = 0
        confidence = LoopBpmConfidence()(zeroAudio, bpmEstimate)
        self.assertEquals(expectedConfidence, confidence)

    def testConstantInput(self):
        onesAudio = ones(100)
        bpmEstimate = 0
        expectedConfidence = 0
        confidence = LoopBpmConfidence()(onesAudio, bpmEstimate)
        self.assertEquals(expectedConfidence, confidence)

suite = allTests(TestLoopBpmConfidence)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

