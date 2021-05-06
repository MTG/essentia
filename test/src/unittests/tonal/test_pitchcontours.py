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
import random
import numpy as np

# recommended processing chain default parameters
defaultHopSize = 128
defaultFrameSize = 2048
defaultBinResolution = 10
defaultMinDuration  = 100
defaultPeakDistributionThreshold = 0.9
defaultPeakFrameThreshold  = 0.9
defaultPitchContinuity = 27.5625
defaultSampleRate = 44100
defaultTimeContinuity = 100

testPeakBins = [[156. , 36.], [154. , 34.], [153., 33.], [152., 32.]]
testPeakSaliences = [[0.05581059, 0.04464847], [0.07149016, 0.05719213], [0.08664553, 0.06931642], [0.10018335, 0.08014668]]

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

    def testDuration(self):
        # simple test for the duration output with small populated frames
        nonEmptyPeakBins = testPeakBins
        nonEmptyPeakSaliences = testPeakSaliences
        theHopSize= 2 * defaultHopSize
        _,  _, _, duration = PitchContours(hopSize=theHopSize)(nonEmptyPeakBins, nonEmptyPeakSaliences)
        calculatedDuration = (2 * theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)
        
        theHopSize= 4 * defaultHopSize
        _,  _, _, duration = PitchContours(hopSize=theHopSize)(nonEmptyPeakBins, nonEmptyPeakSaliences)
        calculatedDuration = (2 * theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)

        theHopSize= 8 * defaultHopSize
        _,  _, _, duration = PitchContours(hopSize=theHopSize)(nonEmptyPeakBins, nonEmptyPeakSaliences)
        calculatedDuration = (2 * theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)                       
    
    def testEmpty(self):
        emptyPeakBins = []
        emptyPeakSaliences = []
        bins, saliences, startTimes, duration = PitchContours()(emptyPeakBins, emptyPeakSaliences)

        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        calculatedDuration = (2*defaultHopSize)/defaultSampleRate
        self.assertEqual(duration, 0)

    def testEmptyColumns(self):
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        theHopSize= 2*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        calculatedDuration = (2*theHopSize)/defaultSampleRate
        # TODO This is a huge relative error threshold. Why is that?
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    #256 frames x single zero-salience peak at the same cent bin in each frame.
    def testSingleZeroSaliencePeak(self):
        peakBins = array(zeros([1, 256]))
        peakSaliences = array(zeros([1, 256]))
        # TODO  It is unclear why we need to change the default hop size recommended for pitch contour estimation.
        theHopSize = 1*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(peakBins, peakSaliences)
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        calculatedDuration = (2*theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    #  We want to have different random frequency peaks at each frame.
    #  Since a random generator is used bins, saliences, startTimes will be different every time
    #  but the duration will always be deterministic for a given number of frames.
    #  The assert checks on random generated outputs are boundary checks/sanity checks
    def testVariousSaliencePeaks(self):
        testPeakBins=[]
        f1 = random.randrange(50, 151, 1)
        f2 = random.randrange(500, 1001, 1)
        # We conisder 600 frames
        index= 0
        while index < 600:
            testPeakBins.append([f1 , f2])
            index+=1
        testPeakSaliences = np.random.random((600, 2))
        bins, saliences, startTimes, duration = PitchContours(hopSize=defaultHopSize)(testPeakBins, testPeakSaliences)
        calculatedDuration = 600*(defaultHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)
        self.assertEqual(len(bins),len(saliences))
        self.assertEqual(len(bins),len(startTimes))
        self.assertGreater(len(bins),0)

    def testUnequalInputs(self):
        peakBins = [zeros(4096), zeros(4096)]
        peakSaliences = [zeros(1024), zeros(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))

    def testRegressionSynthetic(self):
        # Use synthetic audio for Regression Test. This keeps NPY files size low.
        # First, create our algorithms:
        hopSize = defaultHopSize
        frameSize = defaultFrameSize
        sampleRate = defaultSampleRate
        guessUnvoiced = True

        run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
        run_spectrum = Spectrum(size=frameSize * 4)
        run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                           maxFrequency=20000,
                                           maxPeaks=100,
                                           sampleRate=sampleRate,
                                           magnitudeThreshold=0,
                                           orderBy="magnitude")
        run_pitch_salience_function = PitchSalienceFunction()
        run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks()
        run_pitch_contours = PitchContours(hopSize=hopSize)

        signalSize = frameSize * 10
        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc
        c3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 130.81 * 2*math.pi)
        d3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 146.83 * 2*math.pi)
        e3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 164.81 * 2*math.pi)
        f3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 174.61 * 2*math.pi)
        g3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 196.00 * 2*math.pi)                                
        a3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 220.00 * 2*math.pi)
        b3 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 246.94 * 2*math.pi)
        c4 = 0.5 * numpy.sin((array(range(signalSize))/44100.) * 261.63 * 2*math.pi)
    
        # This signal is a "major scale ladder"
        scale = concatenate([c3, d3, e3, f3, g3, a3, b3, c4])

        # Now we are ready to start processing.
        # 1. Load audio and pass it through the equal-loudness filter
        audio = EqualLoudness()(scale)
        calculatedContourBins = []
        calculatedContourSaliences = []
        calculatedContourStartTimes = []
        calculatedDuration =[]
        # ... and create a Pool
        pool = Pool();

        # 2. Cut audio into frames and compute for each frame:
        #    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            frame = run_windowing(frame)
            spectrum = run_spectrum(frame)
            peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
            
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
            
            pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
            pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

        # 3. Now, as we have gathered the required per-frame data, we can feed it to the contour 
        #    tracking and melody detection algorithms:
        contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
                pool['allframes_salience_peaks_bins'],
                pool['allframes_salience_peaks_saliences'])


        save('calculatedcontour_bins_scale0.npy', contours_bins[0])
        save('calculatedcontour_saliences_scale0.npy', contours_saliences[0])
        save('calculatedcontour_bins_scale1.npy', contours_bins[1])
        save('calculatedcontour_saliences_scale1.npy', contours_saliences[1])
        save('calculatedcontour_startTimes_scale.npy', contours_start_times)

        expectedBins0 = load(join(filedir(), 'pitchsalience/calculatedcontour_bins_scale0.npy'))
        expectedSaliences0 = load(join(filedir(), 'pitchsalience/calculatedcontour_saliences_scale0.npy'))
        expectedBins1 = load(join(filedir(), 'pitchsalience/calculatedcontour_bins_scale1.npy'))
        expectedSaliences1 = load(join(filedir(), 'pitchsalience/calculatedcontour_saliences_scale1.npy'))
        expectedStartTimes = load(join(filedir(), 'pitchsalience/calculatedcontour_startTimes_scale.npy'))
   
        expectedBinsList0 = expectedBins0.tolist()
        expectedSaliencesList0 = expectedSaliences0.tolist()
        expectedBinsList1 = expectedBins1.tolist()
        expectedSaliencesList1 = expectedSaliences1.tolist()
        expectedStartTimesList = expectedStartTimes.tolist()
        expectedDuration = len(audio)/(44100)

        self.assertAlmostEqualVectorFixedPrecision(expectedBinsList0, contours_bins[0], 8)
        self.assertAlmostEqualVectorFixedPrecision(expectedSaliencesList0, contours_saliences[0], 8)
        self.assertAlmostEqualVectorFixedPrecision(expectedBinsList1, contours_bins[1], 8)
        self.assertAlmostEqualVectorFixedPrecision(expectedSaliencesList1, contours_saliences[1], 8)
        self.assertAlmostEqualVectorFixedPrecision(expectedStartTimesList, contours_start_times, 8)                        
        self.assertAlmostEqual( expectedDuration, duration, 8)                           

suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
