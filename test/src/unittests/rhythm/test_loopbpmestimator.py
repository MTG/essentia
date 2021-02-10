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

class TestLoopBpmEstimator(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(LoopBpmEstimator(), { 'confidenceThreshold': -1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()        
      
        expectedEstimate= 125
        estimate = LoopBpmEstimator(confidenceThreshold=0.95)(audio)            
        self.assertAlmostEqual( expectedEstimate ,estimate,0.05) 
        
        # Test for upperbound of confidence Threshold
        estimate = LoopBpmEstimator(confidenceThreshold=1)(audio)     
        self.assertAlmostEqual( expectedEstimate ,estimate,0.05)

        # Test for lowerbound of confidence Threshold
        estimate = LoopBpmEstimator(confidenceThreshold=0)(audio)    
        self.assertAlmostEqual( expectedEstimate ,estimate,0.05)
        
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

