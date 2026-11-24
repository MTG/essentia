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
    
    # Test for all the values above the boundary limits.
    def testInvalidParam(self):
        self.assertConfigureFails(PitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(PitchMelodia(), {'binResolution': 0})        
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': -1})        
        self.assertConfigureFails(PitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(PitchMelodia(), {'frameSize': 0})        
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': 2})        
        self.assertConfigureFails(PitchMelodia(), {'hopSize': 0})               
        self.assertConfigureFails(PitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': 0})        
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': 2})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'minDuration': 0})        
        self.assertConfigureFails(PitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(PitchMelodia(), {'minFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': 0})        
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': 2.1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(PitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(PitchMelodia(), {'referenceFrequency': 0})             
        self.assertConfigureFails(PitchMelodia(), {'referenceFrequency': -1})              
        self.assertConfigureFails(PitchMelodia(), {'sampleRate': 0})        
        self.assertConfigureFails(PitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(PitchMelodia(), {'timeContinuity': 0})
        self.assertConfigureFails(PitchMelodia(), {'timeContinuity': -1})

    def testEmpty(self):
        pitch, confidence = PitchMelodia()([])
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(confidence, [])

    def testZeros(self):
        signal = zeros(1024)
        pitch, confidence = PitchMelodia()(signal)
        self.assertEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqualVector(confidence, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testOnes(self):
        signal = ones(1024)
        pitch, confidence = PitchMelodia()(signal)   
        self.assertEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqualVector(confidence, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testSinglePeak(self):
        # generate test signal: sine 110Hz @44100kHz
        frameSize = 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 110 * 2*math.pi)
        pm = PitchMelodia()
        pitch, confidence = pm(signal)

        rounded_pitch = [round(num) for num in pitch]
        rounded_confidence= [round(num,1) for num in confidence]
        
        expectedPitch = repeat(110.0,len(pitch)-10)        
        expectedConfidence = repeat(0.5,len(pitch)-10)

        # Zero pad the expected values
        paddedExpectedPitch = pad(expectedPitch, (5,5), 'constant')      
        paddedExpectedConfidence = pad(expectedConfidence, (5,5), 'constant')             
        # Small Patch
        # In the boundary between zeros and valuies of 110.Hz
        # there is an observed value of 109 Hz.
        paddedExpectedPitch[5] = 109.0 
        
        self.assertAlmostEqualVectorFixedPrecision(paddedExpectedPitch, rounded_pitch,1)
        self.assertAlmostEqualVectorFixedPrecision(paddedExpectedConfidence, rounded_confidence, 1)        

    def testSinglePeakNonDefaultBR(self):   
        # Same as above, but tweaking the Bin Resolution to ensure output length is consistant
        # Larger bin resolution reduces the number of non zero values in salience function
        # generate test signal: sine 110Hz @44100kHz
        binResolution = 40
        frameSize = 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 110 * 2*math.pi)
        pm = PitchMelodia(binResolution=binResolution)
        pitch, confidence = pm(signal)
        rounded_pitch = [round(num) for num in pitch]
        rounded_confidence= [round(num,1) for num in confidence]        
        expectedPitch = repeat(110.0,len(pitch)-10)        
        expectedConfidence = repeat(0.5,len(pitch)-10)

        # Zero pad the expected values
        paddedExpectedPitch = pad(expectedPitch, (5,5), 'constant')      
        paddedExpectedConfidence = pad(expectedConfidence, (5,5), 'constant')             
        
        self.assertAlmostEqualVectorFixedPrecision(paddedExpectedPitch, rounded_pitch,1)
        self.assertAlmostEqualVectorFixedPrecision(paddedExpectedConfidence, rounded_confidence, 1)        

    def testSinglePeakLowCompression(self):
        # generate test signal: sine 110Hz @44100kHz
        frameSize = 4096
        signalSize = 10 * frameSize
        signal = 0.2 * numpy.sin((array(range(signalSize))/44100.) * 110 * 2*math.pi)
        pm = PitchMelodia(magnitudeCompression=0.0001)

        pitch, confidence = pm(signal)

        expectedPitch = repeat(110. ,len(pitch)-13)        
        expectedConfidence = repeat(1. ,len(confidence))      
        
        self.assertAlmostEqualVectorFixedPrecision(expectedPitch, pitch[6:len(pitch)-7],1)
        self.assertAlmostEqualVectorFixedPrecision(expectedConfidence, confidence, 1)        


    def testSinglePeakLowestMagThreshold(self):
        # Provide a single input peak with a unit magnitude at the reference frequency,
        # generate test signal: sine 110Hz @44100kHz
        frameSize = 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 55 * 2*math.pi)
        pm = PitchMelodia(magnitudeThreshold=0)
        pitch, confidence = pm(signal)
        expectedPitch = repeat(0. ,len(pitch))        
        expectedConfidence = repeat(0. ,len(confidence))              

        self.assertAlmostEqualVectorFixedPrecision(expectedPitch, pitch,1)
        self.assertAlmostEqualVectorFixedPrecision(expectedConfidence, confidence, 1)      

    def testTwoPeaksHarmonics(self):
        frameSize= 4096
        signalSize = 10 * frameSize          
        signal_55Hz = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 55 * 2*math.pi)
        signal_110Hz = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 110 * 2*math.pi)
        signal = signal_55Hz + signal_110Hz
        pm = PitchMelodia()
        pitch, confidence = pm(signal)  

        # Check sample points
        self.assertAlmostEqual(pitch[50], 110.0, 1)
        self.assertAlmostEqualFixedPrecision(confidence[50], 0.5, 1)      


        rounded_confidence= [round(num,1) for num in confidence]        
    
        expectedConfidence = repeat(0.5,len(pitch)-8)

        # Zero pad the expected values
        paddedExpectedConfidence = pad(expectedConfidence, (4,4), 'constant')                             
        self.assertAlmostEqualVectorFixedPrecision(paddedExpectedConfidence, rounded_confidence, 1)     
        # FIXME add a test to see that the range of values in the pitch are between 109 and 113
        # pitch also has 4 zeros at beginning and end.

    def testDifferentPeaks(self):  
        frameSize= 4096
        signalSize = 10 * frameSize          
        signal_55Hz = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 55 * 2*math.pi)
        signal_85Hz = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 85 * 2*math.pi)
        signal = signal_55Hz+signal_85Hz
        pm = PitchMelodia()
        pitch, confidence = pm(signal)
        index = 83
        # Do a boundary check at the first bin location
        while index < 83+46:
            self.assertGreater(pitch[index], 55)  
            self.assertLess(pitch[index], 85)  
            index += 1        

    #  Similar unit tests to pitch salience function
    #  This is really a regression test. Reference values and locations are from previous runs.
    def testBelowReferenceFrequency1(self):        
        frameSize= 4096
        signalSize = 10 * frameSize        
        f = 50.0
        signal_50Hz = 1.5 * numpy.sin((array(range(signalSize))/44100.) * f * 2*math.pi)
        binResolution = 10  # defaut value           
        fiveOctaveFullRange = 6000 # 6000 cents covers 5 octaves
        outputLength  = int(fiveOctaveFullRange/binResolution)   
        expectedPitchSalience = zeros(outputLength)
        pitch, confidence  = PitchMelodia()(signal_50Hz)
        index = 10
        # Do an approximation  check at the first bin location
        while index < 30:
            self.assertAlmostEqual(pitch[index], 2*f, 1)
            index += 1                

    # This is really a regression test. Reference values and locations are from previous runs.
    def testBelowReferenceFrequency2(self):
        frameSize = 4096
        signalSize = 10 * frameSize       
        referenceFrequency= 40 
        f = 30.0        
        signal_30Hz = 1.5 * numpy.sin((array(range(signalSize))/44100.) * f * 2*math.pi)
        binResolution = 10 # defaut value             
        fiveOctaveFullRange = 6000 # 6000 cents covers 5 octaves
        outputLength  = int(fiveOctaveFullRange/binResolution)           
        expectedPitchSalience = zeros(outputLength)
        pitch, confidence  = PitchMelodia(referenceFrequency=40)(signal_30Hz)              

        index = 10
        # Do an approximation  check at the first bin location
        while index < 30:
            self.assertAlmostEqual(pitch[index], 2*referenceFrequency, 1)
            index += 1                
        pitch, confidence  = PitchMelodia()(signal_30Hz)              

        index = 10
        # Do an approximation  check at the first bin location
        while index < 30:
            self.assertAlmostEqual(pitch[index], 2*referenceFrequency, 1)
            index += 1                

    def testSinglePeakAboveMaxBin(self):
        frameSize= 4096
        signalSize = 10 * frameSize   
        binResolution = 10 # defaut value             
        fiveOctaveFullRange = 6000 # 6000 cents covers 5 octaves
        outputLength  = int(fiveOctaveFullRange/binResolution)              
        # Choose a sample set of frequencies and magnitude vectors of unequal length
        # Lets test at 800 Hz
        signal_800Hz = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 800 * 2*math.pi)
        pitch, confidence   =  PitchMelodia(binResolution=5)(signal_800Hz)       
        index = 5
        while index < len(pitch)-5:
            self.assertAlmostEqualFixedPrecision(round(pitch[index]), 800, 1)  
            self.assertGreater(confidence[index], 0.4)  
            index += 1        

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)
       
        #This code stores reference values in a file for later loading.
        # save('pitchmelodiapitch.npy', pitch)             
        # save('pitchmelodiaconfidence.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch.npy'))
        expectedPitchMelodiaPitch = loadedPitchMelodiaPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitchMelodiaPitch, 2)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence.npy'))
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 2)

    def testRegressionEqualLoud(self):
        # Since EqualLoudness() is typically used to preprocess an audio  prior
        # to passing through with PitchMelodia, this real scenario is regression tested here.
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        eq = EqualLoudness()
        eqAudio = eq(audio)
        pitch, pitchConfidence = pm(eqAudio)

        #This code stores reference values in a file for later loading.
        # save('pitchmelodiapitch_eqloud.npy', pitch)             
        # save('pitchmelodiaconfidence_eqloud.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch_eqloud.npy'))
        expectedPitchMelodiaPitch = loadedPitchMelodiaPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitchMelodiaPitch, 2)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence_eqloud.npy'))
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 2)

    def testRegressionSyntheticInput(self):
        # generate test signal concatenating major scale notes.
        defaultSampleRate = 44100
        frameSize = 2048
        signalSize = 5 * frameSize
        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc

        c3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 130.81 * 2*math.pi)
        d3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 146.83 * 2*math.pi)
        e3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 164.81 * 2*math.pi)
        f3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 174.61 * 2*math.pi)
        g3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 196.00 * 2*math.pi)                                
        a4 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 220.00 * 2*math.pi)
        b4 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 246.94 * 2*math.pi)
                
        # This signal is a "major scale ladder"
        scale = concatenate([c3, d3, e3, f3, g3, a4, b4])

        pm = PitchMelodia()
        pitch, confidence = pm(scale)

        numPitchSamples = len(pitch)
        numSinglePitchSamples = int(numPitchSamples / 8)
        midPointOffset =  int(numSinglePitchSamples / 2)

        # On each step of the "SCALE LADDER" we take the step mid point.
        # We calculate array index mid point to allow checking the estimated pitch.
        midpointC3 = midPointOffset
        midpointD3 = int(1 * numSinglePitchSamples) + midPointOffset
        midpointE3 = int(2 * numSinglePitchSamples) + midPointOffset
        midpointF3 = int(3 * numSinglePitchSamples) + midPointOffset
        midpointG3 = int(4 * numSinglePitchSamples) + midPointOffset
        midpointA3 = int(5 * numSinglePitchSamples) + midPointOffset        
        midpointB3 = int(6 * numSinglePitchSamples) + midPointOffset                                    
             
        # Check rounded freq. values of notes at middle points 
        # They should align within +/- 1,2 Hz.
        self.assertEqual(round(pitch[midpointC3]), 131, 0)
        self.assertEqual(round(pitch[midpointD3]), 147, 0)
        self.assertEqual(round(pitch[midpointE3]), 165, 0)
        self.assertEqual(round(pitch[midpointF3]), 173, 0)
        self.assertEqual(round(pitch[midpointG3]), 174, 0)
        self.assertEqual(round(pitch[midpointA3]), 196, 0)
        self.assertEqual(round(pitch[midpointB3]), 220, 0)

        # Perform a test for a range of values for the notes at the beginning 
        # and the end of the scale (C3 and B4)
        expectedC3=repeat(130.8,68)
        expectedB4=repeat(246.9,62)
        self.assertAlmostEqualVectorFixedPrecision(expectedC3 ,pitch[5:73],1)   
        self.assertAlmostEqualVectorFixedPrecision(expectedB4, pitch[487:549],1)

suite = allTests(TestPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)