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


expectedSalienceBins_250_500 = [373. ,253., 532., 183., 414., 462., 564., 133., 514., 
    341.,  95., 475.,  63., 444., 314.,  36., 292.,  13., 394., 272., 355., 237., 222., 
    208., 195., 172., 158.,] 

expectedSalienceValues_250_500 =  [1.5773721, 0.9571765,  0.7999999,  0.6839805,  
    0.6712224,  0.6399999,  0.6399997,  0.52685237, 0.5119997,  0.41678187, 
    0.40960002, 0.4095998,  0.32768002, 0.32767984, 0.30070633, 0.26214403, 
    0.24626765, 0.20971522, 0.2097151,  0.20028552, 0.13421766, 0.12412241, 
    0.08589935, 0.06871948, 0.05497558, 0.03518437, 0.03012587]

expectedSalienceBins_1_2_3p5k = [562., 477., 334., 405., 444., 524., 363., 287., 
    368., 595., 324., 145., 215., 427., 306., 237., 388., 203., 260., 95., 184., 
    578., 510., 56., 169., 24., 119.,  11.] 

expectedSalienceValues_1_2_3p5k = [3.8184898, 3.7930138, 3.2797484,  3.2550404, 3.132056,
    3.1184888, 2.8864121,  2.8510342,  2.8067179,  2.6923957, 2.297534, 2.2706633,
    2.23549, 2.0100977, 1.9537883, 1.8793114, 1.4680803, 1.430501,
    1.3932227, 1.2944084, 1.1557142, 1.048053, 1.0399925, 0.9279995,
    0.9250192, 0.86216104, 0.69234276, 0.17134354]

class TestPitchSalienceFunctionPeaks(TestCase):
  
    def testInvalidParam(self):
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'binResolution': -1})
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'minFrequency': -1})
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'referenceFrequency': -1})                

    def testZero(self):
        self.assertEqual(len(PitchSalienceFunctionPeaks()(zeros(1024))), 2)
       
    def testEmpty(self): 
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunctionPeaks()([]))
    
    def testOne(self):
        self.assertEqual(len(PitchSalienceFunctionPeaks()(ones(1024))), 2)

    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        audio = audio[0:136000] # make sure an even size
        # Get the frequencies and magnitudes of the spectral peaks
        freq_speaks, mag_speaks= SpectralPeaks()(audio)
        # Start with default params
        pitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)
        calculatedPitchSalienceBins,calculatedPitchSalienceValues = PitchSalienceFunctionPeaks()(pitchSalience)

        """
        This code stores reference values in a file for later loading.        
        save('pitchsaliencebins.npy', calculatedPitchSalienceBins)             
        save('pitchsaliencevalues.npy', calculatedPitchSalienceValues)                
        """

        # Reference samples are loaded as expected values
        loadedPitchSalienceBins = load(join(filedir(), 'pitchsalience/pitchsaliencebins.npy'))
        loadedPitchSalienceValues = load(join(filedir(), 'pitchsalience/pitchsaliencevalues.npy')) 

        expectedPitchSalienceBins = loadedPitchSalienceBins.tolist() 
        expectedPitchSalienceValues = loadedPitchSalienceValues.tolist() 

        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceBins, expectedPitchSalienceBins, 8)
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceValues, expectedPitchSalienceValues, 8)

    def testSyntheticCase(self):
        sr = 44100
        frameSize = 1024
        hopSize = 512

        size = 1*sr
        sine1 = [sin(2.0*pi*250.0*i/sr) for i in range(size)]
        sine2 = [sin(2.0*pi*500.0*i/sr) for i in range(size)]
        sine3 = [sin(2.0*pi*1000.0*i/sr) for i in range(size)]
        sine4 = [sin(2.0*pi*2000.0*i/sr) for i in range(size)]
        sine5 = [sin(2.0*pi*3500.0*i/sr) for i in range(size)]
        fc = FrameCutter()

        frame1 = fc(sine1)
        frame2 = fc(sine2)
        frame3 = fc(sine3)
        frame4 = fc(sine4)
        frame5 = fc(sine5)                
        audio1 = frame1 + frame2
        audio2 = frame3 + frame4 + frame5

        # Define PSFG with default params
        psf = PitchSalienceFunction()

        # Get the frequencies and magnitudes of the spectral peaks
        freq_speaks, mag_speaks = SpectralPeaks()(audio1)

        pitchSalience = psf(freq_speaks,mag_speaks)
        calculatedPitchSalienceBins,calculatedPitchSalienceValues = PitchSalienceFunctionPeaks()(pitchSalience)

        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceBins, expectedSalienceBins_250_500, 6)
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceValues, expectedSalienceValues_250_500, 6)        

        # Get the frequencies and magnitudes of the spectral peaks
        freq_speaks, mag_speaks = SpectralPeaks()(audio2)
        pitchSalience = psf(freq_speaks,mag_speaks)
        calculatedPitchSalienceBins,calculatedPitchSalienceValues = PitchSalienceFunctionPeaks()(pitchSalience)

        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceBins, expectedSalienceBins_1_2_3p5k, 6)
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceValues, expectedSalienceValues_1_2_3p5k, 6)           

suite = allTests(TestPitchSalienceFunctionPeaks)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
