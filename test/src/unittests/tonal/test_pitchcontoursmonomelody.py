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

class TestPitchContoursMonoMelody(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(PitchContoursMonoMelody(), {'binResolution': -1})
        self.assertConfigureFails(PitchContoursMonoMelody(), {'filterIterations': -1})
        self.assertConfigureFails(PitchContoursMonoMelody(), {'hopSize': -1})
        self.assertConfigureFails(PitchContoursMonoMelody(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchContoursMonoMelody(), {'minFrequency': -1})
        self.assertConfigureFails(PitchContoursMonoMelody(), {'referenceFrequency': -1})
        self.assertConfigureFails(PitchContoursMonoMelody(), {'sampleRate': -1})
        
    def testZero(self):
        bins = [zeros(1024), zeros(1024)] 
        saliences = [zeros(1024), zeros(1024)]
        startTimes = zeros(1024)
        duration =  0.0
        pitch, pitchConfidence = PitchContoursMonoMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])

    def testEmpty(self):
        bins = [[],[]]
        saliences = [[],[]]
        startTimes = [] 
        duration =  0.0
        self.assertRaises(RuntimeError, lambda: PitchContoursMonoMelody()(bins, saliences, startTimes, duration))


    def testOnes(self):
        bins = [ones(1024), ones(1024)] 
        saliences = [ones(1024), ones(1024)]
        startTimes = ones(1024)
        duration =  0.0
        pitch, pitchConfidence = PitchContoursMonoMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])

    def testUnequalInputs(self):
        peakBins = [ones(4096), ones(4096)]
        peakSaliences = [ones(1024), ones(1024)]
        startTimes = ones(1024)
        duration =  0.0        
        pitch, pitchConfidence = PitchContoursMonoMelody()( peakBins, peakSaliences, startTimes, duration)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])
        
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
        pcm = PitchContoursMonoMelody()

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
        pitch, pitchConfidence = pcm(bins, saliences, startTimes, duration)   
        #This code stores reference values in a file for later loading.
        save('pitchcontoursmonomelodypitch.npy', pitch)
        save('pitchcontoursmonomelodyconfidence.npy', pitchConfidence)
       
        loadedPitch = load(join(filedir(), 'pitchcontours/pitchcontoursmonomelodypitch.npy'))        
        loadedPitchConfidence = load(join(filedir(), 'pitchcontours/pitchcontoursmonomelodyconfidence.npy'))
        expectedPitch = loadedPitch.tolist() 
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitch,2)
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence,2)
        

suite = allTests(TestPitchContoursMonoMelody)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
