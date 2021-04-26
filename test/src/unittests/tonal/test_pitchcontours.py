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
        
    def testEmpty(self):
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        bins, saliences, startTimes, duration = PitchContours()(emptyPeakBins, emptyPeakSaliences)       
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        # Previous tests showed small duration of 0.0058 seconds for zero or empty inputs.        
        self.assertAlmostEqual(duration, 0.0058, 8)

    def testZeros(self):
        bins, saliences, startTimes, duration = PitchContours()(array(zeros([2,256])), array(zeros([2,256])))      
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        # Previous tests showed small duration of 0.0058 seconds for zero or empty inputs.
        self.assertAlmostEqual(duration, 0.0058, 8)

    def testZerosUnequalInputs(self):        
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
        
    def testOnesUnequalInputs(self):
        peakBins = [ones(4096), ones(4096)]
        peakSaliences = [ones(1024), ones(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))

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

        #The reason is that 2D vectors dont reload easily from files, using the method shown below.
        save('pitchcontourstarttimes_synthetic.npy',startTimes) 
        save('pitchcontourbins_synthetic.npy', bins)               
        save('pitchcontoursaliences_synthetic.npy', saliences)

        # Captured from previous runs of pitch contours on the synthesis
        expectedDuration = 0.00290249427780509
        
        loadedPitchContourBins = load(join(filedir(), 'pitchcontours/pitchcontourbins_synthetic.npy'))        
        loadedPitchContourSaliences = load(join(filedir(), 'pitchcontours/pitchcontoursaliences_synthetic.npy'))
        loadedPitchContourStartTimes = load(join(filedir(), 'pitchcontours/pitchcontourstarttimes_synthetic.npy'))

        expectedPitchContourBins = loadedPitchContourBins.tolist() 
        expectedPitchContourSaliences = loadedPitchContourSaliences.tolist() 
        expectedPitchContourStartTimes = loadedPitchContourStartTimes.tolist() 

        index = 0
        while index<len(expectedPitchContourBins):
           self.assertAlmostEqualVectorFixedPrecision(expectedPitchContourBins[index], bins[index], 8)
           index+=1         

        index = 0
        while index<len(expectedPitchContourSaliences):
           self.assertAlmostEqualVectorFixedPrecision(expectedPitchContourSaliences[index], saliences[index], 8)
           index+=1         

        self.assertAlmostEqualVectorFixedPrecision(expectedPitchContourStartTimes,startTimes, 8)        
        self.assertAlmostEqual(expectedDuration,duration, 8)             

 
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()        

        # Declare the algorithmns to be called (6 in total)
        psf = PitchSalienceFunction()
        psfp = PitchSalienceFunctionPeaks()
        w = Windowing(type='hann', normalized=False)
        spectrum = Spectrum()
        spectralpeaks = SpectralPeaks()
        pc = PitchContours()

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

        #This code stores reference values in a file for later loading.
        save('pitchcontourstarttimes_real.npy',startTimes) 
        save('pitchcontourbins_real.npy',bins[0])               
        save('pitchcontoursaliences_real.npy', saliences[0])

        # Captured from previous runs of pitch contours on "vignesh" audio
        expectedDuration = 0.7720634937286377    
        loadedPitchContourStartTimes = load(join(filedir(), 'pitchcontours/pitchcontourstarttimes_real.npy'))
        loadedPitchContourBins = load(join(filedir(), 'pitchcontours/pitchcontourbins_real.npy'))        
        loadedPitchContourSaliences = load(join(filedir(), 'pitchcontours/pitchcontoursaliences_real.npy'))

        expectedPitchContourStartTimes = loadedPitchContourStartTimes.tolist() 
        expectedPitchContourBins = loadedPitchContourBins.tolist() 
        expectedPitchContourSaliences = loadedPitchContourSaliences.tolist() 
       
        # FIXME: Extend this to all of the columns not just the first one
        # using a "while loop" on the code below yields the following error:
        # /usr/local/lib/python3.8/dist-packages/numpy/core/_asarray.py:136: VisibleDeprecationWarning: 
        # Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or 
        # ndarrays with different lengths or shapes) is deprecated. 
        # If you meant to do this, you must specify 'dtype=object' when creating the ndarray
        # return array(a, dtype, copy=False, order=order, subok=True)
        #
        # The current workaround this FIXME is to do a full "2D regression test" using all the bins 
        # and saliences in "testRegressionSynthetic"

        self.assertAlmostEqualVectorFixedPrecision(expectedPitchContourBins, bins[0], 8)
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchContourSaliences, saliences[0], 8)
        self.assertAlmostEqualVectorFixedPrecision(expectedPitchContourStartTimes,startTimes, 8)          
        self.assertEqual( duration, expectedDuration)
  
suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

