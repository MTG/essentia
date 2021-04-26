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

class TestPitchMelodia(TestCase):
     
    def testZero(self):
        signal = zeros(1024)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(PitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(PitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': 2})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(PitchMelodia(), {'minFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': 2.1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(PitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(PitchMelodia(), {'referenceFrequency': -1})             
        self.assertConfigureFails(PitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(PitchMelodia(), {'timeContinuity': -1})

    def testOnes(self):
        signal = ones(1024)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testEmpty(self):
        pitch, confidence = PitchMelodia()([])
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(confidence, [])

    def testARealCase(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)
       
        #This code stores reference values in a file for later loading.
        save('pitchmelodiapitch.npy', pitch)             
        save('pitchmelodiaconfidence.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch.npy'))
        expectedPitchMelodiaPitch = loadedPitchMelodiaPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitchMelodiaPitch, 8)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence.npy'))
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 8)

    def testARealCaseEqualLoud(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        eq = EqualLoudness()
        eqAudio = eq(audio)
        pitch, pitchConfidence = pm(eqAudio)

        #This code stores reference values in a file for later loading.
        save('pitchmelodiapitch_eqloud.npy', pitch)             
        save('pitchmelodiaconfidence_eqloud.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch_eqloud.npy'))
        expectedPitchMelodiaPitch = loadedPitchMelodiaPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitchMelodiaPitch, 8)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence_eqloud.npy'))
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 8)

    def test110Hz(self):
        # generate test signal: sine 110Hz @44100kHz
        frameSize= 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 110 * 2*math.pi)
        pm = PitchMelodia()
        pitch, _ = pm(signal)
        index= int(len(pitch)/2) # Halfway point in pitch array
        self.assertAlmostEqual(pitch[50], 110.0, 10)

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

        pm = PitchMelodia()
        pitch, confidence = pm(scale)

        numPitchSamples = len(pitch)
        numSinglePitchSamples = int(numPitchSamples/8)
        midPointOffset =  int(numSinglePitchSamples/2)

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
        # TODO: Check why returned confidence value is low (0.49)
        self.assertAlmostEqual(pitch[midpointC3], 130.81, 10)
        self.assertAlmostEqual(pitch[midpointD3], 146.83, 10)
        self.assertAlmostEqual(pitch[midpointE3], 164.81, 10)
        self.assertAlmostEqual(pitch[midpointF3], 174.61, 10)
        self.assertAlmostEqual(pitch[midpointG3], 196.00, 10)
        self.assertAlmostEqual(pitch[midpointA3], 220.00, 10)
        self.assertAlmostEqual(pitch[midpointB3], 246.94, 10)
        self.assertAlmostEqual(pitch[midpointC4], 261.63, 10)

suite = allTests(TestPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
