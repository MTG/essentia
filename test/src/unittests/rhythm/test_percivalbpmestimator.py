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


from essentia_test import *
from essentia.standard import MonoLoader, PercivalBpmEstimator

class TestPercivalBpmEstimator(TestCase):

    def testInvalidParam(self):
        # testing that output is valid (not NaN nor inf nor negative time values)
        self.assertConfigureFails(PercivalBpmEstimator(), { 'frameSize': -1 })
        self.assertConfigureFails(PercivalBpmEstimator(), { 'frameSizeOSS': -1 })
        self.assertConfigureFails(PercivalBpmEstimator(), { 'hopSize': -1 })
        self.assertConfigureFails(PercivalBpmEstimator(), { 'hopSizeOSS': -1 })
        self.assertConfigureFails(PercivalBpmEstimator(), { 'maxBPM': -1 })
        self.assertConfigureFails(PercivalBpmEstimator(), { 'minBPM': -1 })
        self.assertConfigureFails(PercivalBpmEstimator(), { 'sampleRate': -1 })
                
    # PercivalBpmEstimator() is called inside LoopBpmEstimator().
    # LoopBpmEstimator() also considers "confidence" as a parameter.
    # Regression tests are made on current performance observations
    # which return an accurate value to 1 place of decimal.
    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        expectedEstimate = 125
        estimate = PercivalBpmEstimator()(audio)            
  
        # Tolerance tuned to 0.1 based on emperical test resulting in BPM = 125.28
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 
        # prints 125.28408813476562

        # Define Markers for significant meaningful subsections 
        # to give proportional relationship with audio length.
        # Similart strategy used for  LoopBpmEstimator()-
        len90 = int(0.9*len(audio))   # End point for 90% of loop
        len75 = int(0.75*len(audio))  # 75% point    
        len50 = int(0.5*len(audio))   # mid point                       

        # If any future changes break these asserts,
        # then this will indicates something in algorithm has changed.
        expectedEstimate = 124.9
        estimate = PercivalBpmEstimator()(audio[0:len90])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 
        # prints 124.90558624267578
        estimate = PercivalBpmEstimator()(audio[5000:len75])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 
        # prints 124.90558624267578        
        estimate = PercivalBpmEstimator()(audio[0:len50])                
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 
        # prints 124.90558624267578

    def testSilentEdge(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        bpmEstimate = 125
        lenSilence = 30000 # N.B The beat period is 21168 samples for 125 bpm @ 44.1k samp. rate
        silentAudio = zeros(lenSilence)
        expectedEstimate = 124.9

        # Test addition of non-musical silence before the loop starts
        # The length is not a beat period,
        # case 1: there is non-musical* silence before the loop starts
        signal1 = numpy.append(silentAudio, audio)
        estimate = PercivalBpmEstimator()(signal1)     
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # case 2: there is non-musical silence after the loop ends        
        signal2  = numpy.append(audio,silentAudio)
        estimate = PercivalBpmEstimator()(signal2)
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1)

        # case 3: there is non-musical silence at both ends
        signal3 = numpy.append(signal1, silentAudio)
        estimate = PercivalBpmEstimator()(signal3)      
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1)

    # In the previous test, the length of the silence is independent of the sample length
    # This test examines response to adding silences of lengths equal to a multiple of the beat period.
    def testExactAudioLengthMatch(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        bpmEstimate = 125
        beatPeriod = 21168 # N.B The beat period is 21168 samples for 125 bpm @ 44.1k samp. rate
        silentAudio = zeros(beatPeriod)
        expectedEstimate = 124.9        

        # Add non-musical silence to the beginning of the audio
        signal1 = numpy.append(silentAudio, audio)
        estimate = PercivalBpmEstimator()(signal1)  
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # Add non-musical silence to the end of the audio
        signal2 = numpy.append(audio, silentAudio)
        estimate = PercivalBpmEstimator()(signal2)        
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1) 

        # Concatenate silence at both ends
        signal3 = numpy.append(signal1, silentAudio)
        estimate = PercivalBpmEstimator()(signal3)                    
        self.assertAlmostEqual(expectedEstimate, estimate, 0.1)

    # A series of assert checks on the BPM estimator for empty, zero or constant signals.
    # PercivalBpmEstimator uses the percival estimator internally.
    # The runtime errors have their origin in that algorithm.
    def testEmpty(self):
        emptyAudio = []
        self.assertRaises(RuntimeError, lambda: PercivalBpmEstimator()(emptyAudio))

    def testZero(self):
        beatPeriod = 21168 # N.B The beat period is 21168 samples for 125 bpm @ 44.1k samp. rate
        zeroAudio = zeros(beatPeriod)
        estimate = PercivalBpmEstimator()(zeroAudio)                
        self.assertEqual(estimate, 0.0) 

    def testConstantInput(self):
        beatPeriod = 21168 # N.B The beat period is 21168 samples for 125 bpm @ 44.1k samp. rate
        onesAudio = ones(beatPeriod)        
        estimate = PercivalBpmEstimator()(onesAudio)  
        # The observed BPM is also  104.4 for constant input of ones.
        self.assertAlmostEqual(estimate, 104.40341,8) 

        constantInput = [0.5 for i in range(21168)]
        estimate = PercivalBpmEstimator()(constantInput)      
        # The observed BPM is also around 104.4 for another constant input value, 0.5        
        self.assertAlmostEqual(estimate, 104.40341,8) 

        constantInput = [0.5 for i in range(21168)]
        #Repeat test but, tweak a config. parameters out of its default value.
        estimate = PercivalBpmEstimator(maxBPM=60)(constantInput)          
        self.assertAlmostEqual(estimate, 104.40341,8) 

suite = allTests(TestPercivalBpmEstimator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
