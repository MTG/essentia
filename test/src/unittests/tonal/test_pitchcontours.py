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

class TestPitchContours(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(PitchContours(), {'binResolution': -1})
        self.assertConfigureFails(PitchContours(), {'hopSize': -1})
        self.assertConfigureFails(PitchContours(), {'minDuration': -1})
        self.assertConfigureFails(PitchContours(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchContours(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchContours(), {'pitchContinuity': -1})
        self.assertConfigureFails(PitchContours(), {'sampleRate': -1})
        self.assertConfigureFails(PitchContours(), {'timeContinuity': -1})
        
    def testZero(self):
        peakBins = [zeros(4096),zeros(4096)]
        peakSaliences = [zeros(4096),zeros(4096)]
        bins, saliences, startTimes, duration = PitchContours()(peakBins, peakSaliences)
        
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        self.assertAlmostEqual(duration, 0.0058, 3)
        
        peakBins = [zeros(4096), zeros(4096)]
        peakSaliences = [zeros(1024), zeros(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))

    def testEmpty(self):
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        #self.assertComputeFails(PitchContours()(emptyPeakBins, emptyPeakSaliences))
    
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        audio = audio[0:136000] # make sure an even size
        #pitch, pitchConfidence = PitchMelodia(audio)
        freq_speaks, mag_speaks= SpectralPeaks()(audio)
        # Start with default params
        psf = PitchSalienceFunction()
        salienceFunction = psf(freq_speaks,mag_speaks)
                               
        psfp = PitchSalienceFunctionPeaks()
        bins,values = psfp(salienceFunction)
        print(bins)
        print(values)
        pc = PitchContours()
        
        cbins, csaliences, cstartTimes, cduration = pc([bins,bins],[values,values])
        print(cbins)
        print(csaliences)
        print(cstartTimes)
        print(cduration)
        #This code stores reference values in a file for later loading.

        #save('pitchsaliencefunction.npy', calculatedPitchSalience)             
        #save('pitchsaliencefunction.npy', calculatedPitchSalience)             
        #save('pitchsaliencefunction.npy', calculatedPitchSalience)             
        #save('pitchsaliencefunction.npy', calculatedPitchSalience)             
        # Reference samples are loaded as expected values
        #loadedPitchSalience = load(join(filedir(), 'pitchsalience/pitchsaliencefunction.npy'))
        #expectedPitchSalience = loadedPitchSalience.tolist() 
        #self.assertAlmostEqualVectorFixedPrecision(calculatedPitchSalience, expectedPitchSalience,2)
    
suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
