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

    def testZero(self):
        freqs = zeros(1024)
        mags  = zeros(1024)
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunction()(freqs, mags))

    def testEmpty(self): 
        self.assertEqualVector(PitchSalienceFunction()([], []), zeros(600))

    def testOne(self):        
        self.assertEqualVector(PitchSalienceFunction()(ones(1024), ones(1024)), zeros(600))

    # Provide a single input peak with a unit magnitude at the reference frequency, and 
    # validate that the output salience function has only one non-zero element at the first bin.
    def testSinglePeak(self):
        freq_speaks = [55] # length 1
        mag_speaks = [1] # length 4
        # For a single frequency 55 Hz with unitary  a,plitude the 10 non zero salience function values are the following
        expectedSalienceFunction = [1.0000000e+00, 9.7552824e-01, 9.0450847e-01, 7.9389262e-01, 6.5450847e-01,
         5.0000000e-01, 3.4549147e-01, 2.0610739e-01, 9.5491491e-02, 2.4471754e-02 ]    

        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)    
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience[:10], expectedSalienceFunction,6)


    def testDifferentPeaks(self):
        freq_speaks = [55, 85] # length 1
        mag_speaks = [1, 1] # length 4
        calculatedPitchSalience1 = PitchSalienceFunction()(freq_speaks,mag_speaks)
        save('pitchsaliencefunction_diffpeaks1.npy', calculatedPitchSalience1)                               

        mag_speaks = [0.5, 2] # length 4
        calculatedPitchSalience2 = PitchSalienceFunction()(freq_speaks,mag_speaks)
        save('pitchsaliencefunction_diffpeaks2.npy', calculatedPitchSalience2)                               
        # Reference samples are loaded as expected values
        # Reference samples are loaded as expected values
        loadedPitchSalience1 = load(join(filedir(), 'pitchsalience/pitchsaliencefunction_diffpeaks1.npy'))
        loadedPitchSalience2 = load(join(filedir(), 'pitchsalience/pitchsaliencefunction_diffpeaks2.npy'))

        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience1, loadedPitchSalience1, 6)
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience2, loadedPitchSalience2, 6)



    def testBelowReferenceFrequency1(self):
        freq_speaks = [50] # length 1
        mag_speaks = [1] # length 4
        expectedPitchSalience = zeros(600)
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)        
        self.assertEqualVector(calculatedPitchSalience, expectedPitchSalience)
    

    # Provide multiple duplicate peaks at the reference frequency.
    def testBelowReferenceFrequency2(self):
        freq_speaks = [30] # length 1
        mag_speaks = [1] # length 4
        expectedPitchSalience = zeros(600)
        calculatedPitchSalience = PitchSalienceFunction(referenceFrequency=40)(freq_speaks,mag_speaks)        
        self.assertEqualVector(calculatedPitchSalience, expectedPitchSalience)      
        
    # Provide multiple duplicate peaks at the reference frequency.

    # Provide a single input peak below the reference frequency, 
    # so that the result is an empty pitch salience function

    # How does a single input peak above the maximum bin affect the salience function? Y
    #ou would see the activations at the subharmonics of the input frequency, that fall within the 6000-cent frequency range.
    #def testMaximumBin(self):

    def testUnequalInputs(self):
        # Choose a sample set of frequencies and magnitude vectors of unqual length
        freqs = [250, 500, 1000, 2000, 3500] # length 5
        mags = [0.5, 0.5, 0.5, 0.5] # length 4
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freqs, mags))

    def testNegativeMagnitureTest(self):
        freqs = [250, 500, 1000] # length 3
        mags = [1, -1, 1] # length 3
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freqs, mags))

    def testRegressionTest(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()

        # Get the frequencies and magnitudes of the spectral peaks
        run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
        run_spectrum = Spectrum(size=frameSize * 4)
        run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                           maxFrequency=20000,
                                           maxPeaks=100,
                                           sampleRate=44100,
                                           magnitudeThreshold=0,
                                           orderBy="magnitude")

        run_pitch_salience_function = PitchSalienceFunction()


        #  Cut audio into frames and compute for each frame:
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            frame = run_windowing(frame)
            spectrum = run_spectrum(frame)
            peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
            
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
        
        #This code stores reference values in a file for later loading.        
        save('pitchsaliencefunction.npy', salience)                               
        # Reference samples are loaded as expected values
        loadedPitchSalience = load(join(filedir(), 'pitchsalience/pitchsaliencefunction.npy'))
        self.assertAlmostEqualVectorFixedPrecision(salience, loadedPitchSalience, 6)


suite = allTests(TestPitchSalienceFunction)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
