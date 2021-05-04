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

class TestPitchSalienceFunction(TestCase):
  
    def testInvalidParam(self):
        self.assertConfigureFails(PitchSalienceFunction(), {'binResolution': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'harmonicWeight': 2})
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeCompression': 2})        
        self.assertConfigureFails(PitchSalienceFunction(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(PitchSalienceFunction(), {'numberHarmonics': 0})
        self.assertConfigureFails(PitchSalienceFunction(), {'numberHarmonics': -1})        
        self.assertConfigureFails(PitchSalienceFunction(), {'referenceFrequency': -1})

    def testEmpty(self): 
        self.assertEqualVector(PitchSalienceFunction()([], []), zeros(600))

    # Provide a single input peak with a unit magnitude at the reference frequency, and 
    # validate that the output salience function has only one non-zero element at the first bin.
    def testSinglePeak(self):
        expectedPitchSalience = [1.0000000e+00, 9.7552824e-01, 9.0450847e-01, 7.9389262e-01, 6.5450847e-01,
        5.0000000e-01, 3.4549147e-01, 2.0610739e-01, 9.5491491e-02, 2.4471754e-02, 3.7493994e-33]
        freq_speaks = [55] 
        mag_speaks = [1] 
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)
        # Check the first 11 elements are staring from vlaue "1" and degrading downwards
        # as define in the expectedPitchSalience
        self.assertAlmostEqualVector(calculatedPitchSalience[:11],expectedPitchSalience,6)
        # Check remaining elemenmts are zeros
        self.assertEqualVector(calculatedPitchSalience[12:600],zeros(600-12))

    def test3Peaks(self):
        freq_speaks = [55, 100, 340] 
        mag_speaks = [1, 1, 1] 

        # For a single frequency 55 Hz with unitary  amplitude the 10 non zero salience function values are the following
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)    
  
        save('calculatedPitchSalience_test3Peaks.npy', calculatedPitchSalience)
        self.assertAlmostEqual(calculatedPitchSalience[0], 1.16384, 3)        
        self.assertEqual(len(calculatedPitchSalience), 600)        
        # Reference samples are loaded as expected values
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_test3Peaks.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience,6)

    def test3PeaksHw0(self):
        freq_speaks = [55, 100, 340] 
        mag_speaks = [1, 1, 1] 
        # For a single frequency 55 Hz with unitary  amplitude the 10 non zero salience function values are the following
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=0)(freq_speaks, mag_speaks)            
        self.assertEqual(calculatedPitchSalience[0], 1)
        self.assertEqual(len(calculatedPitchSalience), 600)
                
    def test3PeaksHw1(self):
        freq_speaks = [55, 100, 340] 
        mag_speaks = [1, 1, 1] 
        # For a single frequency 55 Hz with unitary  amplitude the 10 non zero salience function values are the following
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=1)(freq_speaks, mag_speaks) 
        save('calculatedPitchSalience_test3PeaksHw1.npy', calculatedPitchSalience)

        # Reference samples are loaded as expected values
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_test3PeaksHw1.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience,6)
        
    def testDifferentPeaks(self):
        freq_speaks = [55, 85] 
        mag_speaks = [1, 1] 
        calculatedPitchSalience1 = PitchSalienceFunction()(freq_speaks,mag_speaks)             
        save('calculatedPitchSalience_testDifferentPeaks1.npy', calculatedPitchSalience1)

        mag_speaks = [0.5, 2] 
        calculatedPitchSalience2 = PitchSalienceFunction()(freq_speaks,mag_speaks)
        save('calculatedPitchSalience_testDifferentPeaks2.npy', calculatedPitchSalience2)
 
        self.assertAlmostEqual(calculatedPitchSalience1[0], 0.5, 3)        
        self.assertAlmostEqual(calculatedPitchSalience2[0], 1.5, 3)

        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_testDifferentPeaks1.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience1,6)
        
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_testDifferentPeaks2.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList, calculatedPitchSalience2,6)
        

    def testBelowReferenceFrequency1(self):
        freq_speaks = [50] 
        mag_speaks = [1] 
        expectedPitchSalience = zeros(600)
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)            
        self.assertEqualVector(calculatedPitchSalience, expectedPitchSalience)

    def testBelowReferenceFrequency2(self):
        freq_speaks = [30] 
        mag_speaks = [1] 
        expectedPitchSalience = zeros(600)
        calculatedPitchSalience = PitchSalienceFunction(referenceFrequency=40)(freq_speaks,mag_speaks)        
        self.assertEqualVector(calculatedPitchSalience, expectedPitchSalience)      

    def testMustContainPostiveFreq(self):
        # Throw in a zero Freq to see what happens. 
        freqs = [0, 250, 400, 1300, 2200, 3300] # length 6
        mags = [1, 1, 1, 1, 1, 1] # length 6 
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunction()(freqs, mags))

    def testUnequalInputs(self):
        # Choose a sample set of frequencies and magnitude vectors of unqual length
        freqs = [250, 400, 1300, 2200, 3300] # length 5
        mags = [1, 1, 1, 1] # length 4
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freqs, mags))

    def testNegativeMagnitudeTest(self):
        freqs = [250, 500, 1000] # length 3
        mags = [1, -1, 1] # length 3
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freqs, mags))

    def testRegressionTest(self):
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frameSize = 2048
        sampleRate = 44100
        guessUnvoiced = True
        hopSize = 512

        #truncate the audio to take 0.5 sec(keep npy file size low)
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
        # 1. pass it through the equal-loudness filter
        audio = EqualLoudness()(audio)
        calculatedPitchSalience = []

        # 2. Cut audio into frames and compute for each frame:
        #    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            frame = run_windowing(frame)
            spectrum = run_spectrum(frame)
            peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            calculatedPitchSalience.append(salience)

        save('calculatedPitchSalience_vignesh.npy', calculatedPitchSalience)
        expectedPitchSalience = load(join(filedir(), 'pitchsalience/calculatedPitchSalience_vignesh.npy'))
        expectedPitchSalienceList = expectedPitchSalience.tolist()

        index=0
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            self.assertAlmostEqualVectorFixedPrecision(expectedPitchSalienceList[index], calculatedPitchSalience[index], 6)              
            index+=1        

    def testRegressionSynthetic(self):
        # Use synthetic audio for Regression Test. This keeps NPY files size low.      
        # First, create our algorithms:
        hopSize = 128
        frameSize = 2048
        sampleRate = 44100
        guessUnvoiced = True

        run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
        run_spectrum = Spectrum(size=frameSize * 4)
        run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                           maxFrequency=20000,
                                           maxPeaks=100,
                                           sampleRate=sampleRate,
                                           magnitudeThreshold=0,
                                           orderBy="magnitude")
        run_pitch_salience_function = PitchSalienceFunction()
        run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks()
        run_pitch_contours = PitchContours(hopSize=hopSize)
        run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
                                                        hopSize=hopSize)
        signalSize = frameSize * 10
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

        # Now we are ready to start processing.
        # 1. Load audio and pass it through the equal-loudness filter
        audio = EqualLoudness()(scale)

        # 2. Cut audio into frames and compute for each frame:
        #    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            frame = run_windowing(frame)
            spectrum = run_spectrum(frame)
            peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
                
        expectedBins = [270., 150. ,80. ,30.]
        expectedPeaks =  [0.09777679, 0.07822143, 0.06257715, 0.05006172]
        self.assertAlmostEqualVectorFixedPrecision(expectedBins, salience_peaks_bins, 6)
        self.assertAlmostEqualVectorFixedPrecision(expectedPeaks, salience_peaks_saliences, 6)

suite = allTests(TestPitchSalienceFunction)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
