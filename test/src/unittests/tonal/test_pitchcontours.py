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
        
    def testZeros(self):
        bins, saliences, startTimes, duration = PitchContours()(array(zeros([2,256])), array(zeros([2,256])))      
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        self.assertAlmostEqual(duration, 0.0058, 3)

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

    def testEmpty(self):
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        bins, saliences, startTimes, duration = PitchContours()(emptyPeakBins, emptyPeakSaliences)       
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        self.assertAlmostEqual(duration, 0.0058, 3)
            
    def testRegression(self):
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
        #FIXME.  Only the first columns of bins and saliences are saved here for later comparison.
        #The reason is that 2D vectors dont reload easily from files, using the method shown below.
        save('pitchcountourbins.npy', bins[0])
        save('pitchcountoursaliences.npy', saliences[0])

        # Captured from pervious runs of pitch contours on "vignesh" audio
        expectedStartTimes = [0.02321995, 0.5543764,  0.22349206]
        expectedDuration = 0.7720634937286377
        
        loadedPitchContourBins = load(join(filedir(), 'pitchcontours/pitchcountourbins.npy'))        
        loadedPitchContourSaliences = load(join(filedir(), 'pitchcontours/pitchcountoursaliences.npy'))
        expectedPitchContourBins = loadedPitchContourBins.tolist() 
        expectedPitchContourSaliences = loadedPitchContourSaliences.tolist() 
        self.assertEqualVector(bins[0], expectedPitchContourBins)
        self.assertEqualVector(saliences[0], expectedPitchContourSaliences)
        self.assertAlmostEqualVectorFixedPrecision(startTimes, expectedStartTimes, 5)
        self.assertEqual( duration, expectedDuration)

suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

