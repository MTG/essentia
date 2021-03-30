#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

    def testZero(self):
        signal = zeros(256)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(PitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(PitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(PitchMelodia(), {'guessUnvoiced': -2})
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': 2})

        self.assertConfigureFails(PitchMelodia(), {'magnitudeThreshold': -1})

        self.assertConfigureFails(PitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': 0})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': 2})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(PitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(PitchMelodia(), {'referenceFrequency': -1})             
        self.assertConfigureFails(PitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(PitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(PitchMelodia(), {'timeContinuity': -1})



        
        peakBins = [zeros(4096), zeros(4096)]
        peakSaliences = [zeros(1024), zeros(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))

    def testOnes(self):
        peakBins = [ones(4096),ones(4096)]
        peakSaliences = [ones(4096),ones(4096)]
        bins, saliences, startTimes, duration = PitchContours()(peakBins, peakSaliences)

        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        self.assertAlmostEqual(duration, 0.0058, 3)
        
    def testUnequalInputs(self):
        peakBins = [ones(4096), ones(4096)]
        peakSaliences = [ones(1024), ones(1024)]
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
        audio1 = audio[0:136000] # make sure an even size
        audio2 = audio[136001:(2*136000)] # make sure an even size
        #pitch, pitchConfidence = PitchMelodia(audio)

        psf = PitchSalienceFunction()
        freq_speaks1, mag_speaks1= SpectralPeaks()(audio1)
        freq_speaks2, mag_speaks2= SpectralPeaks()(audio2)

        # Start with default params
        salienceFunction1 = psf(freq_speaks1,mag_speaks1)                               
        psfp = PitchSalienceFunctionPeaks()
        bins1,values1 = psfp(salienceFunction1)
        bins2= print(bins1/2)        
        values2 = print(values1/2)        

		b= [bins1,bins2]
        v= [values1,values2]
        b2d = numpy.array([numpy.array(bi) for bi in b])
        v2d = numpy.array([numpy.array(vi) for vi in v])
        


        pc = PitchContours()
        
        cbins, csaliences, cstartTimes, cduration = pc(b2d,v2d)
        print("cbins")
        print(cbins)
        print("saliences")
        print(csaliences)
        print("startTimes")
        print(cstartTimes)        
        print("cduration")
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


suite = allTests(TestPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
