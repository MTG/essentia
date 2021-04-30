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
import random

# recommended processing chain default parameters
defaultHopSize = 128
defaultFrameSize = 2048
defaultBinResolution = 10
defaultMinDuration  = 100
defaultPeakDistributionThreshold = 0.9
defaultPeakFrameThreshold  = 0.9
defaultPitchContinuity = 27.5625
defaultSampleRate = 44100
defaultTimeContinuity = 100

class TestPitchContours(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(PitchContours(), {'binResolution': -1})
        self.assertConfigureFails(PitchContours(), {'hopSize': -1})
        self.assertConfigureFails(PitchContours(), {'minDuration': -1})
        self.assertConfigureFails(PitchContours(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchContours(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchContours(), {'pitchContinuity': -1})
        self.assertConfigureFails(PitchContours(), {'sampleRate': -1})
        self.assertConfigureFails(PitchContours(), {'timeContinuity': -1})

    def testDuration(self):
        # simple test for the duration output. It corresponds to the size of peakBins containing frames.
        emptyPeakBins = []
        emptyPeakSaliences = []
        theHopSize= 2 * defaultHopSize
        _,  _, _, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)       
        calculatedDuration = (2 * theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)
        
        theHopSize= 4 * defaultHopSize        
        _,  _, _, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)               
        calculatedDuration = (2 * theHopSize)/defaultSampleRate        
        self.assertAlmostEqual(duration, calculatedDuration, 8)

        theHopSize= 8 * defaultHopSize        
        _,  _, _, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)               
        calculatedDuration = (2 * theHopSize)/defaultSampleRate        
        self.assertAlmostEqual(duration, calculatedDuration, 8)                        
        
    def testEmpty(self):
        emptyPeakBins = []
        emptyPeakSaliences = []
        bins, saliences, startTimes, duration = PitchContours()(emptyPeakBins, emptyPeakSaliences)       
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        # Previous tests showed small duration of 0.0058 seconds for zero or empty inputs.        
        calculatedDuration = (2*defaultHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    def testEmptyColumns(self):
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        theHopSize= 2*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)       
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        # Previous tests showed small duration of 0.0058 seconds for zero or empty inputs.        
        calculatedDuration = (2*theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    def testZeros(self):
        theHopSize = 4*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(array(zeros([2,256])), array(zeros([2,256])))      
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        # Previous tests showed small duration of 0.0058 seconds for zero or empty inputs.
        calculatedDuration = (2*theHopSize)/defaultSampleRate            
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    #256 frames x single zero-salience peak at the same cent bin in each frame.
    def testSingleZeroSaliencePeak(self):
        peakBins = array(zeros([1, 256]))
        peakSaliences = array(zeros([1, 256]))

        theHopSize = 4*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(peakBins, peakSaliences)      
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        # Previous tests showed small duration of 0.0058 seconds for zero or empty inputs.
        calculatedDuration = (2*theHopSize)/defaultSampleRate            
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    #256 frames x small random number of zero-salience peaks at random cent bins in each frame (use a fixed random seed).
    def testVariousZeroSaliencePeaks(self):
        random.seed(10)
        n = random.randrange(2, 16, 1)    
        peakBins = array(zeros([n,256]))
        peakSaliences = array(zeros([n,256]))

        theHopSize = 4*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(array(zeros([2,256])), array(zeros([2,256])))      
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])

        calculatedDuration = (2*theHopSize)/defaultSampleRate            
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    def testUnequalInputs(self):        
        peakBins = [zeros(4096), zeros(4096)]
        peakSaliences = [zeros(1024), zeros(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))
    
suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

