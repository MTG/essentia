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


class TestMultiPitchMelodia(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(MultiPitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(MultiPitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(MultiPitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'magnitudeCompression': 2})
        self.assertConfigureFails(MultiPitchMelodia(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'minFrequency': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakDistributionThreshold': 2.1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(MultiPitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(MultiPitchMelodia(), {'referenceFrequency': -1})             
        self.assertConfigureFails(MultiPitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(MultiPitchMelodia(), {'timeContinuity': -1})

    def testEmpty(self):
        pitch = MultiPitchMelodia()([])
        self.assertEqualVector(pitch, [])

    def testZero(self):
        signal = zeros(1024)
        pitch = MultiPitchMelodia()(signal)
        self.assertEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testOnes(self):
        signal = ones(1024)
        pitch = MultiPitchMelodia()(signal)
        expectedPitch=[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
        index=0
        self.assertEqual(len(pitch), 9)
        while (index<len(expectedPitch)):
            self.assertEqualVector(pitch[index], expectedPitch[index])
            index+=1

    def testMajorScale(self):
        # generate test signal concatenating major scale notes.
        frameSize= 2048
        signalSize = 5 * frameSize

        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc
        c3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 130.81 * 2*math.pi)
        d3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 146.83 * 2*math.pi)
        e3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 164.81 * 2*math.pi)
        f3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 174.61 * 2*math.pi)
        g3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 196.00 * 2*math.pi)                                
        a3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 220.00 * 2*math.pi)
        b3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 246.94 * 2*math.pi)
        c4 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 261.63 * 2*math.pi)
    
        # This signal is a "major scale ladder"
        scale = concatenate([c3, d3, e3, f3, g3, a3, b3, c4])

        mpm = MultiPitchMelodia()
        pitch = mpm(scale)

        numPitchSamples = len(pitch)
        numSinglePitchSamples = int(numPitchSamples/8)
        midPointOffset =  int(numSinglePitchSamples/2)

        theLen = len(pitch)
        index = 0
        multiArray = []  
        while (index < theLen):
            multiArray.append(pitch[index][0])
            index+=1

        midpointC3 = midPointOffset
        midpointD3 = int(1 * numSinglePitchSamples) + midPointOffset
        midpointE3 = int(2 * numSinglePitchSamples) + midPointOffset
        midpointF3 = int(3 * numSinglePitchSamples) + midPointOffset
        midpointG3 = int(4 * numSinglePitchSamples) + midPointOffset
        midpointA3 = int(5 * numSinglePitchSamples) + midPointOffset        
        midpointB3 = int(6 * numSinglePitchSamples) + midPointOffset
        midpointC4 = int(7 * numSinglePitchSamples) + midPointOffset                   
        
        self.assertAlmostEqualFixedPrecision(multiArray[midpointC3], 130.81, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointD3], 146.83, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointE3], 164.81, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointF3], 174.61, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointG3], 196.00, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointA3], 220.00, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointB3], 246.94, 1)
        self.assertAlmostEqualFixedPrecision(multiArray[midpointC4], 261.63, 1)


suite = allTests(TestMultiPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
