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
from numpy import *

class TestPitchSalienceFunction(TestCase):
  
    def testInvalidParam(self):
        self.assertConfigureFails(PitchSalienceFunction(), {'binResolution': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'binResolution': 0})        
        self.assertConfigureFails(PitchSalienceFunction(), {'binResolution': 101})                
        self.assertConfigureFails(PitchSalienceFunction(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'harmonicWeight': 2})
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeCompression': 0})        
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeCompression': 2})        
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'numberHarmonics': 0})
        self.assertConfigureFails(PitchSalienceFunction(), {'numberHarmonics': -1})        
        self.assertConfigureFails(PitchSalienceFunction(), {'referenceFrequency': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'referenceFrequency': 0})        

    def testEmpty(self): 
        self.assertEqualVector(PitchSalienceFunction()([], []), zeros(600))

    def testSinglePeak(self):        
        # Provide a single input peak with a unit magnitude at the reference frequency, and 
        # validate that the output salience function has only one non-zero element at the first bin.       
        # N.B: default value for bin Resolution is 10.
        freq_speaks = [55] 
        mag_speaks = [1] 
        outputLength  = 600 # calculated by fiveOctaveFullRange/binResolution = 6000/10        
         # Length of the non-zero values for this Pitch Salience = 11
        expectedPitchSalience = [1.0000000e+00, 9.7552824e-01, 9.0450847e-01, 7.9389262e-01, 6.5450847e-01,
        5.0000000e-01, 3.4549147e-01, 2.0610739e-01, 9.5491491e-02, 2.4471754e-02, 3.7493994e-33]
        # Append zeros to expected salience
        expectedPitchSalience += [0] * (outputLength-11)
        
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)
        self.assertEqual(len(calculatedPitchSalience), outputLength)       
        # Check the first 11 elements. The first element has value "1".
        # The next returned 10 non-zero values decreasing in magnitude, should match "expected" above.
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience, 8)

    def testSinglePeakNonDefaultBR(self):   
        # Same as above, but tweaking the Bin Resolution to ensure output length is consistant
        # Larger bin resolution reduces the number of non zero values in salience function
        binResolution = 40
        freq_speaks = [55] 
        mag_speaks = [1] 
        outputLength  = int (6000/binResolution)
        # Length of the non-zero values for this Pitch Salience = 3        
        expectedPitchSalience = [1.0000000e+00, 5.0000000e-01, 3.7493994e-33]
        # Append zeros to expected salience
        expectedPitchSalience += [0] * (outputLength-3)
        
        calculatedPitchSalience = PitchSalienceFunction(binResolution=binResolution)(freq_speaks, mag_speaks)
        self.assertEqual(len(calculatedPitchSalience), outputLength)
        # Check the first 3 elements. The first has value "1".
        # The next returned 3 non-zero values decreasing in magnitude, should match "expected" above.        
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience, 8)

    def testSinglePeakLowCompression(self):
        # Provide a single input peak with a 0.5 magnitude at the reference frequency
        freq_speaks = [55]
        mag_speaks = [0.5]
        outputLength  = 600

        # Low Compression expected values (0.0001)
        expectedLowCompPitchSalience = [9.9993068e-01, 9.7546059e-01, 9.0444577e-01, 7.9383761e-01, 6.5446311e-01,
        4.9996534e-01, 3.4546751e-01, 2.0609310e-01, 9.5484875e-02, 2.4470057e-02, 3.7491393e-33]

        # Default magnitudeCompression expected values
        expectedNormalPitchSalience = [5.0000000e-01, 4.8776412e-01, 4.5225424e-01, 3.9694631e-01, 3.2725424e-01,
        2.5000000e-01, 1.7274573e-01, 1.0305370e-01, 4.7745746e-02, 1.2235877e-02, 1.8746997e-33]
        
        # Append zeros to expected saliences
        expectedLowCompPitchSalience += [0] * (outputLength-11)
        expectedNormalPitchSalience += [0] * (outputLength-11)

        calculatedLowCompPitchSalience = PitchSalienceFunction(magnitudeCompression=0.0001)(freq_speaks, mag_speaks)
        self.assertEqual(len(calculatedLowCompPitchSalience), outputLength)
        self.assertAlmostEqualVectorFixedPrecision(calculatedLowCompPitchSalience, expectedLowCompPitchSalience, 8)

        calculatedNormalPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)
        self.assertEqual(len(calculatedNormalPitchSalience), outputLength)
        self.assertAlmostEqualVectorFixedPrecision(calculatedNormalPitchSalience, expectedNormalPitchSalience, 8)

    def testLowMagThreshold(self):
        freq_speaks = [55, 80, 120] # 3 peaks not harmonic
        mag_speaks = [1, 0.1, 0.2]
        outputLength  = 600

        # This is the expected pitch salience from 1 peak at ampl = 1  (testSinglePeak earlier)
        expectedPitchSalience = [1.0000000e+00, 9.7552824e-01, 9.0450847e-01, 7.9389262e-01, 6.5450847e-01,
        5.0000000e-01, 3.4549147e-01, 2.0610739e-01, 9.5491491e-02, 2.4471754e-02, 3.7493994e-33]

        # Append zeros to expected salience
        expectedPitchSalience += [0] * (outputLength-11)  
        # At magThreshold = 13 or lower, only first peak gets through
        calculatedPitchSalience = PitchSalienceFunction(magnitudeThreshold=13)(freq_speaks, mag_speaks)

        self.assertEqual(len(calculatedPitchSalience), outputLength)       
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience, 8)

    def testTwoPeaksHarmonics(self):
        # Provide a 2 input peaks with a unit magnitude and validate
        # that PSF is different depending on numberHarmonics configuration.
        freq_speaks = [55, 110]
        mag_speaks = [1, 1]
        outputLength  = 600

        calculatedPitchSalience1H = PitchSalienceFunction(numberHarmonics=1)(freq_speaks, mag_speaks)
        self.assertEqual(len(calculatedPitchSalience1H), outputLength)
        calculatedPitchSalience20H = PitchSalienceFunction(numberHarmonics=20)(freq_speaks, mag_speaks)
        self.assertEqual(len(calculatedPitchSalience20H), outputLength)
        
        # Save operation is commented out. Uncomment to tweak parameters orinput to genrate new referencesw when required.
        # save('calculatedPitchSalience_test2PeaksHarmonics1H.npy', calculatedPitchSalience1H)
        # save('calculatedPitchSalience_test2PeaksHarmonics20H.npy', calculatedPitchSalience20H)        
        # Reference samples are loaded as expected values

        expectedPitchSalience1H = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_test2PeaksHarmonics1H.npy'))
        expectedPitchSalienceList1H = expectedPitchSalience1H.tolist()
        expectedPitchSalience20H = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_test2PeaksHarmonics20H.npy'))
        expectedPitchSalienceList20H = expectedPitchSalience20H.tolist()
        # Detailed contents check on returned values (regression check
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience1H, expectedPitchSalienceList1H, 7)
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience20H, expectedPitchSalienceList20H, 7)          
     
    def testDuplicatePeaks(self):
        # Provide multiple duplicate peaks at the reference frequency.    
        freq_speaks = [55, 55, 55] 
        mag_speaks = [1, 1, 1] 
        outputLength  = 600        
        # The same expectedPitchSalience from testSinglePeak test case
        expectedPitchSalience = [3.0000000e+00, 2.9265847e+00, 2.7135253e+00, 2.3816779e+00, 1.9635254e+00,
            1.5000000e+00, 1.0364745e+00, 6.1832219e-01, 2.8647447e-01, 7.3415264e-02, 1.1248198e-32]

        # Append zeros to expected salience
        expectedPitchSalience += [0] * (outputLength-11)
        # For 3 duplicate peaks, the expectedPitchSalience needs to be scaled by a factor of 3
        arrayExpectedPitchSalience = 3*array(expectedPitchSalience)
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks) 
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience, 7)

    def testSinglePeakHw0(self):
        freq_speaks = [55] 
        mag_speaks = [1] 
        outputLength  = 600        
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=0.0)(freq_speaks, mag_speaks)            
        self.assertEqual(calculatedPitchSalience[0], 1)
        self.assertEqualVector(calculatedPitchSalience[1:outputLength], zeros(outputLength-1))
        self.assertEqual(len(calculatedPitchSalience), outputLength)
                
    def testSinglePeakHw1(self):   
        freq_speaks = [55] 
        mag_speaks = [1] 
        outputLength  = 600        
        expectedPitchSalience = [1.0000000e+00, 9.7552824e-01, 9.0450847e-01, 7.9389262e-01, 6.5450847e-01,
        5.0000000e-01, 3.4549147e-01, 2.0610739e-01, 9.5491491e-02, 2.4471754e-02, 3.7493994e-33]
        # Append zeros to expected saliences
        expectedPitchSalience += [0] * (outputLength-11)

        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=1.0)(freq_speaks, mag_speaks)     
        self.assertEqual(len(calculatedPitchSalience), outputLength)        
        # Check the first 11 elements. The first has value "1" 
        # The next 10 values are decreasing in magnitude, then zeros.
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience, 8)

    def test3PeaksHw1(self):
        freq_speaks = [55, 100, 340] 
        mag_speaks = [1, 1, 1] 
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=1.0)(freq_speaks, mag_speaks) 
        # Save operation is commented out. Uncomment to tweak parameters orinput to genrate new referencesw when required.
        # save('calculatedPitchSalience_test3PeaksHw1.npy', calculatedPitchSalience)
        # Reference samples are loaded as expected values
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_test3PeaksHw1.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience, 8)
        
    def testDifferentPeaks(self):
        freq_speaks = [55, 85] 
        mag_speaks = [1, 1] 
        calculatedPitchSalience1 = PitchSalienceFunction()(freq_speaks,mag_speaks)             
        # Save operation is commented out. Uncomment to tweak parameters orinput to genrate new referencesw when required.
        # save('calculatedPitchSalience_testDifferentPeaks1.npy', calculatedPitchSalience1)

        mag_speaks = [0.5, 2] 
        calculatedPitchSalience2 = PitchSalienceFunction()(freq_speaks,mag_speaks)
        # Save operation is commented out. Uncomment to tweak parameters orinput to genrate new referencesw when required.        
        # save('calculatedPitchSalience_testDifferentPeaks2.npy', calculatedPitchSalience2)
        self.assertAlmostEqual(calculatedPitchSalience1[0], 0.5, 3)        
        self.assertAlmostEqual(calculatedPitchSalience2[0], 1.5, 3)

        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_testDifferentPeaks1.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience1, 8)
        
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_testDifferentPeaks2.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience2, 8)
        
    def testBelowReferenceFrequency1(self):
        # Provide a single input peak below the reference frequency, so that the result is an empty pitch 
        # salience function        
        freq_speaks = [50] 
        mag_speaks = [1]     
        outputLength  = 600        
        expectedPitchSalience = zeros(outputLength)
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)            
        self.assertEqualVector(calculatedPitchSalience, expectedPitchSalience)

    def testBelowReferenceFrequency2(self):
        freq_speaks = [30] 
        mag_speaks = [1]            
        outputLength  = 600        
        expectedPitchSalience = zeros(outputLength)
        calculatedPitchSalience = PitchSalienceFunction(referenceFrequency=40)(freq_speaks, mag_speaks)        
        self.assertEqualVector(calculatedPitchSalience, expectedPitchSalience)      

    def testMustContainPostiveFreq(self):
        # Throw in a zero Freq to see what happens. 
        freq_speaks = [0, 250, 400, 1300, 2200, 3300] # length 6
        mag_speaks = [1, 1, 1, 1, 1, 1] # length 6 
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunction()(freq_speaks, mag_speaks))

    def testUnequalInputs(self):
        # Choose a sample set of frequencies and magnitude vectors of unequal length
        freq_speaks = [250, 400, 1300, 2200, 3300] # length 5
        mag_speaks = [1, 1, 1, 1] # length 4
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freq_speaks, mag_speaks))

    def testNegativeMagnitudeTest(self):
        freqs = [250, 500, 1000] # length 3
        mag_speaks = [1, -1, 1] # length 3
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freqs, mag_speaks))

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frameSize = 2048
        sampleRate = 44100
        guessUnvoiced = True
        hopSize = 512

        # 1. Truncate the audio to take 0.5 sec (keep npy file size low)
        audio = audio[:22050]

        run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
        run_spectrum = Spectrum(size=frameSize * 4)
        run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                           maxFrequency=20000,
                                           maxPeaks=100,
                                           sampleRate=sampleRate,
                                           magnitudeThreshold=0,
                                           orderBy="magnitude")
        run_pitch_salience_function = PitchSalienceFunction()
       
        # Now we are ready to start processing.
        # 2. pass it through the equal-loudness filter
        audio = EqualLoudness()(audio)
        calculatedPitchSalience = []

        # 3. Cut audio into frames and compute for each frame:
        #    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            frame = run_windowing(frame)
            spectrum = run_spectrum(frame)
            peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            calculatedPitchSalience.append(salience)

        # Save operation is commented out. Uncomment to tweak parameters orinput to genrate new referencesw when required.
        # save('calculatedPitchSalience_vignesh.npy', calculatedPitchSalience)
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_vignesh.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        
        for i in range(len(calculatedPitchSalience)):
            # Tolerance check to 6 decimal palces (for Linux Subsyst. compatibility.
            self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList[i], calculatedPitchSalience[i], 8)     

    # Test for diverse frequency peaks.
    def test3Peaks(self):
        freq_speaks = [55, 100, 340] 
        mag_speaks = [1, 1, 1] 
        outputLength  = 600        
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)    
        # First check the length of the ouput is 600 
        self.assertEqual(len(calculatedPitchSalience), outputLength)       
        # This test case with diverse frequency values to save ouput to NPY file since the output is more complex.
        # Save operation is commented out. Uncomment to tweak parameters orinput to genrate new referencesw when required.        
        # save('calculatedPitchSalience_test3Peaks.npy', calculatedPitchSalience)
        # Reference samples are loaded as expected values
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_test3Peaks.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience, 8)


suite = allTests(TestPitchSalienceFunction)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
