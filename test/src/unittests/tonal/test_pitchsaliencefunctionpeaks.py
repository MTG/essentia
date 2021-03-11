#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
        """
        save('pitchsaliencebins.npy', calculatedPitchSalienceBins)             
        save('pitchsaliencevalues.npy', calculatedPitchSalienceValues)                
      
        # Reference samples are loaded as expected values
        loadedPitchSalienceBins = load(join(filedir(), 'pitchsalience/pitchsaliencebins.npy'))
        loadedPitchSalienceValues = load(join(filedir(), 'pitchsalience/pitchsaliencevalues.npy')) 

        expectedPitchSalienceBins = loadedPitchSalienceBins.tolist() 
        expectedPitchSalienceValues = loadedPitchSalienceValues.tolist() 

        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceBins, expectedPitchSalienceBins,2)
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalienceValues, expectedPitchSalienceValues,2)        

suite = allTests(TestPitchSalienceFunctionPeaks)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
