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

class TestPitchMelodia(TestCase):

    def testInvalidParam(self):
    # Test for all the values above the boundary limits.    	
        self.assertConfigureFails(PitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(PitchMelodia(), {'binResolution': 0})        
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': -1})        
        self.assertConfigureFails(PitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(PitchMelodia(), {'frameSize': 0})        
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': -1})
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

    def testDefaultParameters(self):
        signal = randn(1024)
        pitch, confidence = PitchMelodia()(signal)
        # Assert that default parameters produce valid outputs
        self.assertIsNotNone(pitch)
        self.assertIsNotNone(confidence)

    def testEmptyInput(self):
        pitch, confidence = PitchMelodia()([])
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(confidence, [])

    def testZerosInput(self):
        signal = zeros(1024)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0.] * 9)  # Use [0.] * 9 for flexibility
        self.assertAlmostEqualVector(confidence, [0.] * 9)

    def testOnesInput(self):
        signal = ones(1024)
        pitch, confidence = PitchMelodia()(signal)   
        self.assertAlmostEqualVector(pitch, [0.] * 9)  # Use [0.] * 9 for flexibility
        self.assertAlmostEqualVector(confidence, [0.] * 9)

    def testCustomParameters(self):
        signal = randn(2048)
        # Use custom parameters
        params = {
            'binResolution': 5,
            'filterIterations': 5,
            'frameSize': 1024,
            'guessUnvoiced': True,
            'harmonicWeight': 0.9,
            'hopSize': 256,
            'magnitudeCompression': 0.5,
            'magnitudeThreshold': 30,
            'maxFrequency': 15000,
            'minDuration': 50,
            'minFrequency': 60,
            'numberHarmonics': 15,
            'peakDistributionThreshold': 1.0,
            'peakFrameThreshold': 0.8,
            'pitchContinuity': 30.0,
            'referenceFrequency': 60,
            'sampleRate': 22050,
            'timeContinuity': 150
        }
        pitch, confidence = PitchMelodia(**params)(signal)
        # Assert that custom parameters produce valid outputs
        self.assertIsNotNone(pitch)
        self.assertIsNotNone(confidence)

    def testInputWithSilence(self):
        signal = concatenate([zeros(512), randn(1024), zeros(512)])
        pitch, confidence = PitchMelodia()(signal)
        # Assert that silent portions don't have pitch information
        self.assertTrue(all(p == 0.0 for p in pitch[:512]))
        self.assertTrue(all(c == 0.0 for c in confidence[:512]))

    def testHighPitchResolution(self):
        signal = randn(1024)
        pitch, confidence = PitchMelodia(binResolution=1)(signal)
        # Assert that using high bin resolution produces valid outputs
        self.assertIsNotNone(pitch)
        self.assertIsNotNone(confidence)
        self.assertEqual(len(pitch), 3)
        self.assertEqual(len(confidence), 3)

    def testRealCase(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)

        # Save reference values for later loading
        save('pitchmelodiapitch.npy', pitch)             
        save('pitchmelodiaconfidence.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch.npy'))
        self.assertAlmostEqualVectorFixedPrecision(pitch, loadedPitchMelodiaPitch.tolist(), 8)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence.npy'))
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, loadedPitchConfidence.tolist(), 8)

    def testRealCaseEqualLoudness(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        eq = EqualLoudness()
        eqAudio = eq(audio)
        pitch, pitchConfidence = pm(eqAudio)

        # Save reference values for later loading
        save('pitchmelodiapitch_eqloud.npy', pitch)             
        save('pitchmelodiaconfidence_eqloud.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch_eqloud.npy'))
        self.assertAlmostEqualVectorFixedPrecision(pitch, loadedPitchMelodiaPitch.tolist(), 8)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence_eqloud.npy'))
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, loadedPitchConfidence.tolist(), 8)

    def test110Hz(self):
        signal = 0.5 * numpy.sin((array(range(10 * 4096))/44100.) * 110 * 2*math.pi)
        pm = PitchMelodia()
        pitch, confidence = pm(signal)
        self.assertAlmostEqual(pitch[50], 110.0, 10)

    def test110HzPeakThresholds(self):
        signal = 0.5 * numpy.sin((array(range(10 * 4096))/44100.) * 110 * 2*math.pi)
        pm_default = PitchMelodia()
        pm_hw0 = PitchMelodia(peakFrameThreshold=0)
        pm_hw1 = PitchMelodia(peakFrameThreshold=1)
        
        pitch_default, confidence_default = pm_default(signal)
        pitch_hw0, confidence_hw0 = pm_hw0(signal)
        pitch_hw1, confidence_hw1 = pm_hw1(signal)

        self.assertAlmostEqual(pitch_default[50], 110.0, 10)
        self.assertAlmostEqual(pitch_hw0[50], 110.0, 10)
        self.assertAlmostEqual(pitch_hw1[50], 110.0, 10)

    def testDifferentPeaks(self):  
        signal_55Hz = 0.5 * numpy.sin((array(range(10 * 4096))/44100.) * 55 * 2*math.pi)
        signal_85Hz = 0.5 * numpy.sin((array(range(10 * 4096))/44100.) * 85 * 2*math.pi)
        signal = signal_55Hz + signal_85Hz
        pm = PitchMelodia()
        pitch, confidence = pm(signal)

        for p in pitch[83:129]:  # Adjusted the range to be more clear
            self.assertGreater(p, 55)
            self.assertLess(p, 85)

    def testBelowReferenceFrequency1(self):        
        signal_50Hz = 1.5 * numpy.sin((array(range(10 * 4096))/44100.) * 50 * 2*math.pi)
        pitch, confidence = PitchMelodia()(signal_50Hz)
        self.assertAlmostEqual(pitch[10], 100.0, 2)

    def testBelowReferenceFrequency2(self):
        signal_30Hz = 1.5 * numpy.sin((array(range(10 * 4096))/44100.) * 30 * 2*math.pi)
        pitch, confidence = PitchMelodia(referenceFrequency=40)(signal_30Hz)
        self.assertAlmostEqual(pitch[10], 60.0, 2)        

suite = allTests(TestPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
