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

class TestPitchContoursMelody(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(PitchContoursMelody(), {'binResolution': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'filterIterations': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'hopSize': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'minFrequency': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'referenceFrequency': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'sampleRate': -1})
        self.assertConfigureFails(PitchContoursMelody(), {'voicingTolerance': -2})
        self.assertConfigureFails(PitchContoursMelody(), {'voicingTolerance': 1.5})


    def testEmpty(self):
        emptyBins = [[],[]]
        emptySaliences = [[],[]]
        pitch, pitchConfidence = PitchContoursMelody()( emptyBins, emptySaliences, [], 0)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])
        #self.assertComputeFails(PitchContoursMelody()(emptyPeakBins, emptyPeakSaliences))
        
    def testZeros(self):
        bins = [zeros(1024), zeros(1024)] 
        saliences = [zeros(1024), zeros(1024)]
        startTimes = zeros(1024)
        duration =  0.0
        pitch, pitchConfidence = PitchContoursMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])

    def testZerosUnequalInputs(self):        
        bins = [zeros(4096), zeros(4096)]
        saliences = [zeros(1024), zeros(1024)]
        startTimes = zeros(512)
        duration =  0.0
        pitch, pitchConfidence = PitchContoursMelody()( bins, saliences, startTimes, duration)    
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])

    def testOnes(self):
        bins = [ones(1024), ones(1024)] 
        saliences = [ones(1024), ones(1024)]
        startTimes = ones(1024)
        duration =  0.0
        pitch, pitchConfidence = PitchContoursMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])

    def testOnesUnequalInputs(self):
        bins = [ones(4096), ones(4096)]
        saliences = [ones(1024), ones(1024)]
        startTimes = ones(1024)
        duration =  0.0        
        pitch, pitchConfidence = PitchContoursMelody()( bins, saliences, startTimes, duration)
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(pitchConfidence, [])

    def testRegressionSynthetic(self):
        # Use synthetic audio for Regression Test. This keeps NPY files size low.     
        sr = 44100
        frameSize = 1024
        hopSize = 512

        size = 1*sr
        sine1 = [sin(2.0*pi*100.0*i/sr) for i in range(size)]
        sine2 = [sin(2.0*pi*1000.0*i/sr) for i in range(size)]
        fc1 = FrameCutter()
        fc2 = FrameCutter()
        frame1 = fc1(sine1)
        frame2 = fc2(sine2)
        audio = frame1+frame2

        psf = PitchSalienceFunction()
        psfp = PitchSalienceFunctionPeaks()
        w = Windowing(type='hann', normalized=False)
        spectrum = Spectrum()
        spectralpeaks = SpectralPeaks()
        pc = PitchContours()
        pcm = PitchContoursMelody()        

        # Populate an array of frame-wise vectors of cent bin values representing each contour
        peakBins = []
        # Populate a frame-wise array of values of salience function peaks
        peakSaliences = []        
        for frame in FrameGenerator(audio, frameSize=1024, hopSize=hopSize,
                                    startFromZero=True):
            freq_peaks, mag_peaks = spectralpeaks(spectrum(w(frame)))
            len_freq_peaks = len(freq_peaks)
            len_mag_peaks = len(mag_peaks)
            
            # Skip the first vector elements that correspond to zero frequency
            freq_peaksp1 = freq_peaks[1:len_freq_peaks]
            # Do the same for mag. vector to keep the lengths the same
            mag_peaksp1 = mag_peaks[1:len_mag_peaks]
            salienceFunction = psf(freq_peaksp1, mag_peaksp1)
            bins, values = psfp(salienceFunction)
            peakBins.append(bins)
            peakSaliences.append(values)

        bins, saliences, startTimes, duration = pc(peakBins, peakSaliences)
        pitch, pitchConfidence = pcm(bins, saliences, startTimes, duration)   
        #This code stores reference values in a file for later loading.
        save('pitchcontoursmelodypitch_synthetic.npy', pitch)
        save('pitchcontoursmelodyconfidence_synthetic.npy', pitchConfidence)
       
        loadedPitch = load(join(filedir(), 'pitchcontoursmelody/pitchcontoursmelodypitch_synthetic.npy'))        
        loadedPitchConfidence = load(join(filedir(), 'pitchcontoursmelody/pitchcontoursmelodyconfidence_synthetic.npy'))
        expectedPitch = loadedPitch.tolist() 
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitch, 8)
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 8)        
        
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
        pcm = PitchContoursMelody()

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
        save('pitchcontoursmelodypitch_real.npy', pitch)
        save('pitchcontoursmelodyconfidence_real.npy', pitchConfidence)
       
        loadedPitch = load(join(filedir(), 'pitchcontoursmelody/pitchcontoursmelodypitch_real.npy'))        
        loadedPitchConfidence = load(join(filedir(), 'pitchcontoursmelody/pitchcontoursmelodyconfidence_real.npy'))
        expectedPitch = loadedPitch.tolist() 
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitch, 8)
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 8)
        
suite = allTests(TestPitchContoursMelody)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
