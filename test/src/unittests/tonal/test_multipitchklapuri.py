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


class TestMultiPitchKlapuri(TestCase):

    def testZero(self):
        signal = zeros(1024)
        pitch = MultiPitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(MultiPitchKlapuri(), {'binResolution': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'frameSize': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'harmonicWeight': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'hopSize': -1})        
        self.assertConfigureFails(MultiPitchKlapuri(), {'magnitudeCompression': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'magnitudeCompression': 2})
        self.assertConfigureFails(MultiPitchKlapuri(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'maxFrequency': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'minFrequency': -1})
        self.assertConfigureFails(MultiPitchKlapuri(), {'numberHarmonics': -1})            
        self.assertConfigureFails(MultiPitchKlapuri(), {'referenceFrequency': -1})             
        self.assertConfigureFails(MultiPitchKlapuri(), {'sampleRate': -1})

    def testOnes(self):
        # FIXME. Need to derive a rational why this output occurs for a constant input
        signal = ones(1024)
        pitch = MultiPitchKlapuri()(signal)
        expectedPitch= [[ 92.498886, 184.99854 ],
            [108.110695, 151.1358  ],
            [108.73698,  151.1358  ],
            [108.73698,  151.1358  ],
            [108.73698,  151.1358  ],
            [108.110695, 151.1358  ],
            [ 92.498886, 184.99854 ]]

        self.assertEqual(len(pitch), 7)
        index=0
        while (index<len(expectedPitch)):
            self.assertAlmostEqualVector(pitch[index], expectedPitch[index],8)
            index+=1

    def testEmpty(self):
        pitch = MultiPitchKlapuri()([])
        self.assertEqualVector(pitch, [])

    def test110Hz(self):
        # generate test signal: sine 110Hz @44100kHz
        frameSize= 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 110 * 2*math.pi)
        mpk = MultiPitchKlapuri()
        pitch = mpk(signal)
        index= int(len(pitch)/2) # Halfway point in pitch array
        self.assertAlmostEqual(pitch[index], 110.0, 10)

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

        mpk = MultiPitchKlapuri()
        pitch = mpk(scale)

        numPitchSamples = len(pitch)
        numSinglePitchSamples = int(numPitchSamples/8)
        midPointOffset =  int(numSinglePitchSamples/2)

        theLen = len(pitch)
        index = 0
        klapArray = []  
        while (index < theLen):
            klapArray.append(pitch[index][0])
            index+=1

        # On each step of the "SCALE LADDER" we take the step mid point.
        # We calculate array index mid point to allow checking the estimated pitch.

        midpointC3 = midPointOffset
        midpointD3 = int(1 * numSinglePitchSamples) + midPointOffset
        midpointE3 = int(2 * numSinglePitchSamples) + midPointOffset
        midpointF3 = int(3 * numSinglePitchSamples) + midPointOffset
        midpointG3 = int(4 * numSinglePitchSamples) + midPointOffset
        midpointA3 = int(5 * numSinglePitchSamples) + midPointOffset        
        midpointB3 = int(6 * numSinglePitchSamples) + midPointOffset
        midpointC4 = int(7 * numSinglePitchSamples) + midPointOffset                                        
             
        # Use high precision (10) for checking synthetic signals
        self.assertAlmostEqual(klapArray[midpointC3], 130.81, 10)
        self.assertAlmostEqual(klapArray[midpointD3], 146.83, 10)
        self.assertAlmostEqual(klapArray[midpointE3], 164.81, 10)
        self.assertAlmostEqual(klapArray[midpointF3], 174.61, 10)
        self.assertAlmostEqual(klapArray[midpointG3], 196.00, 10)
        self.assertAlmostEqual(klapArray[midpointA3], 220.00, 10)
        self.assertAlmostEqual(klapArray[midpointB3], 246.94, 10)
        self.assertAlmostEqual(klapArray[midpointC4], 261.63, 10)

suite = allTests(TestMultiPitchKlapuri)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
