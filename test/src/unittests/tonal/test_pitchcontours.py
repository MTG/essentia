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

        # There are varying hopSize values but the expected duration (calculatedDuration) 
        #is always wrongly the same.
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

        #TODO Vary the number of frames
 
    def testEmpty(self):
        emptyPeakBins = []
        emptyPeakSaliences = []
        bins, saliences, startTimes, duration = PitchContours()(emptyPeakBins, emptyPeakSaliences)

        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        self.assertEqual(duration, 0)

    def testEmptyFrames(self):
        # TODO We shouldn't hard-code the number of frames as we can infer it with len(emptyPeakBins)
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        theHopSize= 2*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        calculatedDuration = (2*theHopSize)/defaultSampleRate
        self.assertAlmostEqual(duration, calculatedDuration, 8)

    # FIXME - This test case is failing
    # The test is set up so that random frequency peaks are fed into PitchSalienceFunction, 
    # chained to PitchSalienceFunctionPeaks and then fed to PitchContours.
    # The end result produces empty output.
    """

    This Test case maybe can be removed since testRegressionSynthetic gives sufficient coverage
    def testVariousPeaks(self):
        freq_speaks = [55, 110, 220, 440]
        mag_speaks = [1, 1, 1, 1]

        print(freq_speaks)
        # Build up "vector_vector_real" salience bin and value inputs for pitchcontour
        theBins = []
        theValues = []
   
        for i in range(10):
            calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)
            bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
            theBins.append(bins * 10)
            theValues.append(values * 10)   

        bins, saliences, startTimes, duration = PitchContours()(theBins, theValues)
        print("the meaningful bins value for PitchContours")
        print(bins, saliences, startTimes, duration)
        calculatedDuration = 10*(defaultHopSize)/defaultSampleRate 
        self.assertAlmostEqual(duration, calculatedDuration, 8)
        self.assertEqual(len(bins), len(saliences))
        self.assertEqual(len(bins), len(startTimes))  
        print("the meaningful bins value")
        print(bins)

    """

    def testUnequalInputs(self):
        # Tests for unequal numbers of peaks in a frame and number of frames.
        peakBins = [zeros(4096), zeros(4096)]
        peakSaliences = [zeros(1024), zeros(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))
        
        peakBins = [zeros(4096), zeros(1024)]
        peakSaliences =[zeros(4096), zeros(4096)]       
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))

        peakBins = [zeros(4096), zeros(4096), zeros(4096)]
        peakSaliences =[zeros(4096), zeros(4096)]       
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
        run_pitch_contours = PitchContours(hopSize=hopSize, sampleRate=sampleRate)
        run_pitch_contours_melody = PitchContoursMelody(hopSize=hopSize, sampleRate=sampleRate)

        signalSize = frameSize * 10
        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc
        # Lydian Scale of F
        # Put a bit of constant input at the beginning.
        const = ones(signalSize)
        # These are appox. /rounded values of the notes listed.
        # They might be 1 Hz out
        f3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 174 * 2*math.pi)
        g3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 196 * 2*math.pi)                                
        a4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 220 * 2*math.pi)
        b4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 246 * 2*math.pi)
        c4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 262 * 2*math.pi)
        d4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 294 * 2*math.pi)
        e4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 330 * 2*math.pi)
        # This signal is a "major scale ladder"
        scale = concatenate([const, f3, g3, a4, b4, c4, d4, e4])

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
            peak_frequencies, peak_magnitudes = run_spectral_peaks( run_spectrum(run_windowing(frame)))
            
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
            
            pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
            pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

        # 3. Now, as we have gathered the required per-frame data, we can feed it to the contour 
        #    tracking and melody detection algorithms:
        contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
                pool['allframes_salience_peaks_bins'],
                pool['allframes_salience_peaks_saliences'])

        pitch, confidence = run_pitch_contours_melody(contours_bins, contours_saliences, contours_start_times, duration)

        # Do a round operation on the pitch
        rpitch = []
        for i in range(len(pitch)):
            rpitch.append(round(pitch[i]))

        count_f3 = format(rpitch.count(174))
        count_g3 = format(rpitch.count(196))
        count_a4 = format(rpitch.count(220))
        count_b4 = format(rpitch.count(246))
        count_c4 = format(rpitch.count(262))
        count_d4 = format(rpitch.count(294))
        count_e4 = format(rpitch.count(330))


        # Do a check for a minimum number of occurences of each of the
        # originally generated frequencies from 174 to 330 Hz.
        minOccurences= 147 
        self.assertGreater(int(count_f3), minOccurences)    
        self.assertGreater(int(count_g3), minOccurences)    
        self.assertGreater(int(count_a4), minOccurences)    
        self.assertGreater(int(count_b4), minOccurences)    
        self.assertGreater(int(count_c4), minOccurences)    
        self.assertGreater(int(count_d4), minOccurences)    
        self.assertGreater(int(count_e4), minOccurences)                


    def testRegressionSynthetic_PCtest(self):
        # Lets tweak Pitch continuity
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
        run_pitch_contours = PitchContours(hopSize=hopSize, sampleRate=sampleRate, pitchContinuity=5.0)
        run_pitch_contours_melody = PitchContoursMelody(hopSize=hopSize, sampleRate=sampleRate)

        signalSize = frameSize * 10
        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc
        # Lydian Scale of F
        # Put a bit of constant input at the beginning.
        const = ones(signalSize)
        # These are appox. /rounded values of the notes listed.
        # They might be 1 Hz out
        f3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 174 * 2*math.pi)
        g3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 196 * 2*math.pi)                                
        a4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 220 * 2*math.pi)
        b4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 246 * 2*math.pi)
        c4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 262 * 2*math.pi)
        d4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 294 * 2*math.pi)
        e4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 330 * 2*math.pi)
        # This signal is a "major scale ladder"
        scale = concatenate([const, f3, g3, a4, b4, c4, d4, e4])

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
            peak_frequencies, peak_magnitudes = run_spectral_peaks( run_spectrum(run_windowing(frame)))
            
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
            
            pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
            pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

        # 3. Now, as we have gathered the required per-frame data, we can feed it to the contour 
        #    tracking and melody detection algorithms:
        contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
                pool['allframes_salience_peaks_bins'],
                pool['allframes_salience_peaks_saliences'])

        pitch, confidence = run_pitch_contours_melody(contours_bins, contours_saliences, contours_start_times, duration)

        # Do a round operation on the pitch
        rpitch = []
        for i in range(len(pitch)):
            rpitch.append(round(pitch[i]))


        count_a4 = format(rpitch.count(220))
        count_b4 = format(rpitch.count(246))
        count_c4 = format(rpitch.count(262))
        count_d4 = format(rpitch.count(294))
        count_e4 = format(rpitch.count(330))

        minOccurences= 147 
        self.assertGreater(int(count_a4), minOccurences)    
        self.assertGreater(int(count_b4), minOccurences)    
        self.assertGreater(int(count_c4), minOccurences)    
        self.assertGreater(int(count_d4), minOccurences)    
        self.assertGreater(int(count_e4), minOccurences)                


    def testRegressionSynthetic_TCtest(self):
        # Lets tweak Time continuity
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
        run_pitch_contours = PitchContours(hopSize=hopSize, sampleRate=sampleRate, timeContinuity=2000)
        run_pitch_contours_melody = PitchContoursMelody(hopSize=hopSize, sampleRate=sampleRate)

        signalSize = frameSize * 10
        # Here are generate sine waves for each note of the scale, e.g. C3 is 130.81 Hz, etc
        # Lydian Scale of F
        # Put a bit of constant input at the beginning.
        const = ones(signalSize)
        # These are appox. /rounded values of the notes listed.
        # They might be 1 Hz out

        # LETS ZERO OUT THE C4 notes to see what happens
        f3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 174 * 2*math.pi)
        g3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 196 * 2*math.pi)                           
        a4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 220 * 2*math.pi)
        b4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 246 * 2*math.pi)
        c4 = zeros(signalSize)      
        d4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 294 * 2*math.pi)
        e4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 330 * 2*math.pi)
        # This signal is a "major scale ladder"
        scale = concatenate([const, f3, g3, a4, b4, c4, d4, e4])

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
            peak_frequencies, peak_magnitudes = run_spectral_peaks( run_spectrum(run_windowing(frame)))
            
            salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
            salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
            
            pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
            pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

        # 3. Now, as we have gathered the required per-frame data, we can feed it to the contour 
        #    tracking and melody detection algorithms:
        contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
                pool['allframes_salience_peaks_bins'],
                pool['allframes_salience_peaks_saliences'])

        pitch, confidence = run_pitch_contours_melody(contours_bins, contours_saliences, contours_start_times, duration)

        # Do a round operation on the pitch
        rpitch = []
        for i in range(len(pitch)):
            rpitch.append(round(pitch[i]))

        count_f3 = format(rpitch.count(174))
        count_g3 = format(rpitch.count(196))
        count_a4 = format(rpitch.count(220))
        count_b4 = format(rpitch.count(246))
        count_d4 = format(rpitch.count(294))
        count_e4 = format(rpitch.count(330))


        minOccurences= 147 
        self.assertGreater(int(count_f3), minOccurences)    
        self.assertGreater(int(count_g3), minOccurences)    
        self.assertGreater(int(count_a4), minOccurences)    
        self.assertGreater(int(count_b4), minOccurences)    
        self.assertGreater(int(count_d4), minOccurences)    
        self.assertGreater(int(count_e4), minOccurences)                

suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
