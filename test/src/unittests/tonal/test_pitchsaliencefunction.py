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

    def testUnequalInputs(self):
        # Choose a sample set of frequencies and magnitude vectors of unqual length
        freqs = [250, 500, 1000, 2000, 3500] # length 5
        mags = [0.5, 0.5, 0.5, 0.5] # length 4
        self.assertRaises(EssentiaException, lambda: PitchSalienceFunction()(freqs, mags))

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
        psf = PitchSalienceFunction()
        calculatedPitchSalience = psf(freq_speaks,mag_speaks)

        """
        This code stores reference values in a file for later loading.
        """
        save('pitchsaliencefunction.npy', calculatedPitchSalience)             
      
        # Reference samples are loaded as expected values
        loadedPitchSalience = load(join(filedir(), 'pitchsalience/pitchsaliencefunction.npy'))
        expectedPitchSalience = loadedPitchSalience.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience,2)

suite = allTests(TestPitchSalienceFunction)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
