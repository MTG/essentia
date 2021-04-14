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

class TestPitchContoursMultiMelody(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(PitchContoursMultiMelody(), {'binResolution': -1})
        self.assertConfigureFails(PitchContoursMultiMelody(), {'filterIterations': -1})
        self.assertConfigureFails(PitchContoursMultiMelody(), {'hopSize': -1})
        self.assertConfigureFails(PitchContoursMultiMelody(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchContoursMultiMelody(), {'minFrequency': -1})
        self.assertConfigureFails(PitchContoursMultiMelody(), {'referenceFrequency': -1})
        self.assertConfigureFails(PitchContoursMultiMelody(), {'sampleRate': -1})

        
    def testZero(self):
        bins = [zeros(1024), zeros(1024)] 
        saliences = [zeros(1024), zeros(1024)]
        startTimes = zeros(1024)
        duration =  0.0
        pitch = PitchContoursMultiMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])


    def testEmpty(self):
        bins = [[],[]]
        saliences = [[],[]]
        startTimes = [] 
        duration =  0.0
        self.assertRaises(RuntimeError, lambda: PitchContoursMultiMelody()(bins, saliences, startTimes, duration))

    def testOnes(self):
        bins = [ones(1024), ones(1024)] 
        saliences = [ones(1024), ones(1024)]
        startTimes = ones(1024)
        duration =  0.0
        pitch = PitchContoursMultiMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])


    def testUnequalInputs(self):
        peakBins = [ones(4096), ones(4096)]
        peakSaliences = [ones(1024), ones(1024)]
        startTimes = ones(1024)
        duration =  0.0        
        pitch = PitchContoursMultiMelody()( peakBins, peakSaliences, startTimes, duration)
        self.assertEqualVector(pitch, [])

        
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()        

        # Declare the algorithmns to be called
        psf = PitchSalienceFunction()
        psfp = PitchSalienceFunctionPeaks()
        w = Windowing(type='hann', normalized=False)
        spectrum = Spectrum()
        spectralpeaks = SpectralPeaks()
        pc = PitchContours()
        pcm = PitchContoursMultiMelody()

        peakBins = []
        peakSaliences = []        
        for frame in FrameGenerator(audio, frameSize=1024, hopSize=hopSize,
                                    startFromZero=True):

            freq_speaks, mag_speaks = SpectralPeaks()(audio)
            # Start with default params         
            calculatedPitchSalience = psf(freq_speaks,mag_speaks)
            freq_peaks, mag_peaks = spectralpeaks(spectrum(w(frame)))
            len_freq_peaks = len(freq_peaks)
            len_mag_peaks = len(mag_peaks)
            freq_peaksp1 = freq_peaks[1:len_freq_peaks]
            mag_peaksp1 = mag_peaks[1:len_mag_peaks]
            salienceFunction = psf(freq_peaksp1, mag_peaksp1)
            bins, values = psfp(salienceFunction)
            peakBins.append(bins)
            peakSaliences.append(values)

        bins, saliences, startTimes, duration = pc(peakBins, peakSaliences)
        pitch = pcm(bins, saliences, startTimes, duration)   
        #This code stores reference values in a file for later loading.
        save('pitchcontoursmultimelodypitch.npy', pitch)
       
        loadedPitch = load(join(filedir(), 'pitchcontours/pitchcontoursmultimelodypitch.npy'))        
        expectedPitch = loadedPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitch,2)

        

suite = allTests(TestPitchContoursMultiMelody)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
