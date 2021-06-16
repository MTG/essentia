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
from essentia import Pool
from essentia import array as e_array
import essentia.standard as estd
import random
import numpy as np
import warnings


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

    def testEmpty(self):
        emptyPeakBins = []
        emptyPeakSaliences = []
        bins, saliences, startTimes, duration = PitchContours()(emptyPeakBins, emptyPeakSaliences)

        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        self.assertEqual(duration, 0)

    def testEmptyFrames(self):
        emptyPeakBins = [[],[]]
        emptyPeakSaliences = [[],[]]
        theHopSize= 2*defaultHopSize
        bins, saliences, startTimes, duration = PitchContours(hopSize=theHopSize)(emptyPeakBins, emptyPeakSaliences)
        self.assertEqualVector(bins, [])
        self.assertEqualVector(saliences, [])
        self.assertEqualVector(startTimes, [])
        calculatedDuration = (len(emptyPeakBins)*theHopSize)/defaultSampleRate
        # FIXME Check precision here
        self.assertAlmostEqualFixedPrecision(duration, calculatedDuration, 2)

    def testUnequalInputs(self):
        # Suite o tests for unequal numbers of peaks in a frame, number of frames,etc.
        peakBins = [zeros(4096), zeros(4096)]
        peakSaliences = [zeros(1024), zeros(1024)]
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))
        
        peakBins = [zeros(4096), zeros(1024)]
        peakSaliences =[zeros(4096), zeros(4096)]       
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))

        peakBins = [zeros(4096), zeros(4096), zeros(4096)]
        peakSaliences =[zeros(4096), zeros(4096)]       
        self.assertRaises(RuntimeError, lambda: PitchContours()(peakBins, peakSaliences))
    
    # Helper function
    def _roundArray(self, pitch):
        rpitch = []
        for i in range(len(pitch)):
            rpitch.append(round(pitch[i]))
        return rpitch

    def testRegressionSynthetic(self):
        # Use synthetic audio for Regression Test.
        #
        # FIXME: Pitch and Time continuity checks are made here. No previous reviews were done on theses.
        # 
        #

        hopSize = defaultHopSize
        frameSize = defaultFrameSize
        sampleRate = defaultSampleRate
        guessUnvoiced = True

        run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
        run_spectrum = Spectrum(size=frameSize * 4)
        run_spectral_peaks = SpectralPeaks(minFrequency=1, maxFrequency=20000,
                                           maxPeaks=100, sampleRate=sampleRate,
                                           magnitudeThreshold=0, orderBy="magnitude")
        run_pitch_salience_function = PitchSalienceFunction()
        run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks()
        run_pitch_contours = PitchContours(hopSize=hopSize, sampleRate=sampleRate)

        signalSize = frameSize * 10
        # Here are generate sine waves for each note of the scale, e.g. c3 is 130.81 Hz, etc
        # Lydian Scale of F
        # Put a bit of constant input at the beginning.
        const = ones(signalSize)
        # These are appox. /rounded values of the notes listed f3, g3...etc
        # They might be (1 or 0.5) Hz out)
        # 7 notes from Lydian scale
        f3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 174 * 2*math.pi)
        g3 = 1 * numpy.sin((array(range(signalSize))/44100.) * 196 * 2*math.pi)                                
        a4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 220 * 2*math.pi)
        b4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 246 * 2*math.pi)
        c4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 262 * 2*math.pi)
        d4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 294 * 2*math.pi)
        e4 = 1 * numpy.sin((array(range(signalSize))/44100.) * 330 * 2*math.pi)

        # a5 to be used for pitch continuity tests
        a5 = 1 * numpy.sin((array(range(signalSize))/44100.) * 440 * 2*math.pi)
        a5plusSmallDelta = 1 * numpy.sin((array(range(signalSize))/44100.) * 445 * 2*math.pi)                                
        a5plusMediumDelta = 1 * numpy.sin((array(range(signalSize))/44100.) * 447 * 2*math.pi)                                

        # Create 150ms gap for time continuity tests
        gap150ms = zeros(15*int(defaultSampleRate/100)) 

        # Create 100ms gap for time continuity tests
        gap100ms = zeros(int(defaultSampleRate/10)) 

        # INPUT AUDIO FOR TEST SUITE 1  
        # This signal is a "lydian scale ladder" with constant values at the beginning-
        scale = concatenate([const, f3, g3, a4, b4, c4, d4, e4])
        
        # INPUT AUDIO FOR TEST SUITE 2   
        # Test signals for time continuity
        limitGapAudio = concatenate([const, f3, gap100ms, f3])
        longGapAudio = concatenate([const, f3, gap150ms, gap150ms,f3])        
    

        # INPUT AUDIO FOR TEST SUITE 3   
        # Test signals for pitch continuity
        sigSmalldelta = concatenate([const, a4, a5plusSmallDelta ])
        sigMediumDelta = concatenate([const, a4, a5plusMediumDelta ])

        #####################################################################################################
        contours_bins, contours_start_times, contour_saliences, duration =  self._extract_pitch_contours(scale, tc=defaultTimeContinuity, pc=defaultPitchContinuity)

        # run the simplified contour selection
        [pitch, pitch_salience] = self.select_contours(
            contours_bins, contour_saliences, contours_start_times, duration)

        # cent to Hz conversion
        pitch = [0. if p == 0
                 else 55. * 2. ** (defaultBinResolution * p / 1200.)
                 for p in pitch]
        pitch = e_array(pitch)
        pitch_salience = e_array(pitch_salience)

        # Do a round operation on the pitch
        rpitch = self._roundArray(pitch)

        # Do a check for a minimum number of occurences of each of the
        # originally generated frequencies from 174 to 330 Hz.
        # Regression test show 147 min. re-occurences.
        minOccurences= 147 
        self.assertGreater(int(format(rpitch.count(174))), minOccurences)    
        self.assertGreater(int(format(rpitch.count(196))), minOccurences)    
        self.assertGreater(int(format(rpitch.count(220))), minOccurences)    
        self.assertGreater(int(format(rpitch.count(246))), minOccurences)    
        self.assertGreater(int(format(rpitch.count(262))), minOccurences)    
        self.assertGreater(int(format(rpitch.count(294))), minOccurences)    
        self.assertGreater(int(format(rpitch.count(330))), minOccurences)        

        # Default parameters for continuity shown explicitly to facilitate tweaking the tests.
        contours_bins, contours_start_times, contour_saliences, duration =  self._extract_pitch_contours(limitGapAudio, tc=defaultTimeContinuity, pc=defaultPitchContinuity)
        # run the simplified contour selection
        [pitch, pitch_salience] = self.select_contours(contours_bins, contour_saliences, contours_start_times, duration)

        # Do a round operation on the pitch
        rpitch = self._roundArray(pitch)

        # Check at least 303 instances of pitch value 199
        self.assertGreater(int(format(rpitch.count(199))), 303)    

        contours_bins, contours_start_times, contour_saliences, duration =  self._extract_pitch_contours(sigSmalldelta, tc=defaultTimeContinuity, pc=defaultPitchContinuity)
        # run the simplified contour selection
        [pitch, pitch_salience] = self.select_contours(contours_bins, contour_saliences, contours_start_times, duration)

        # Do a round operation on the pitch
        rpitch = self._roundArray(pitch)

        # Check at least 156 instances of pitch value 362
        self.assertGreater(int(format(rpitch.count(362))), 156)    

        contours_bins, contours_start_times, contour_saliences, duration =  self._extract_pitch_contours(sigMediumDelta, tc=defaultTimeContinuity, pc=defaultPitchContinuity)
        # run the simplified contour selection
        [pitch, pitch_salience] = self.select_contours(contours_bins, contour_saliences, contours_start_times, duration)

        # Do a round operation on the pitch
        rpitch = self._roundArray(pitch)

        # Check at least 156 instances of pitch value 363
        self.assertGreater(int(format(rpitch.count(363))), 156)    

    # FIXME
    #
    # The following code ais  taken the following source code from the following functions
    # select_contours, _extract_pitch_contours, _join_contours, _remove_overlaps
    # https://github.com/sertansenturk/predominantmelodymakam    
    #
    # This is perhaps overkill, and maybe there is a more compact way to achieve test coverage.
    #
    def select_contours(self, pitch_contours, contour_saliences, start_times,
                        duration):
        sample_rate = defaultSampleRate
        hop_size = defaultHopSize

        # number in samples in the audio
        num_samples = int(ceil((duration * sample_rate) / hop_size))

        # Start points of the contours in samples
        start_samples = [
            int(round(start_times[i] * sample_rate / float(hop_size)))
            for i in range(0, len(start_times))]

        pitch_contours_no_overlap = []
        start_samples_no_overlap = []
        contour_saliences_no_overlap = []
        lens_no_overlap = []
        try:
            # the pitch contours is a list of numpy arrays, parse them starting
            # with the longest contour
            while pitch_contours:  # terminate when all the contours are
                # checked
                # print len(pitchContours)

                # get the lengths of the pitchContours
                lens = [len(k) for k in pitch_contours]

                # find the longest pitch contour
                long_idx = lens.index(max(lens))

                # pop the lists related to the longest pitchContour and append
                # it to the new list
                pitch_contours_no_overlap.append(pitch_contours.pop(long_idx))
                contour_saliences_no_overlap.append(
                    contour_saliences.pop(long_idx))
                start_samples_no_overlap.append(start_samples.pop(long_idx))
                lens_no_overlap.append(lens.pop(long_idx))

                # accumulate the filled samples
                acc_idx = range(start_samples_no_overlap[-1],
                                start_samples_no_overlap[-1] +
                                lens_no_overlap[-1])

                # remove overlaps
                [start_samples, pitch_contours, contour_saliences] = self._remove_overlaps(start_samples, pitch_contours, contour_saliences, lens, acc_idx)
                #[start_samples, pitch_contours, contour_saliences] = self._remove_overlaps(start_samples, pitch_contours, contour_saliences, lens)
        except ValueError:
            # if the audio input is very short such that Essentia returns a
            # single contour as a numpy array (of length 1) of numpy array
            # (of length 1). In this case the while loop fails directly
            # as it tries to check all the truth value of an all pitch values,
            # instead of checking whether the list is empty or not.
            # Here we handle the error in a Pythonic way by simply breaking the
            # loop and assigning the inputs to outputs since a single contour
            # means nothing to filter
            pitch_contours_no_overlap = pitch_contours
            contour_saliences_no_overlap = contour_saliences
            start_samples_no_overlap = start_samples

        pitch, salience = self._join_contours(pitch_contours_no_overlap,
                                              contour_saliences_no_overlap,
                                              start_samples_no_overlap,
                                              num_samples)

        return pitch, salience    

    def _extract_pitch_contours(self, audio, tc, pc):

        # Hann window with x4 zero padding
        run_windowing = estd.Windowing(zeroPadding=3 *defaultFrameSize)
        run_spectrum = estd.Spectrum(size=defaultFrameSize * 4)
        run_spectral_peaks = estd.SpectralPeaks(minFrequency=1,
                                           maxFrequency=20000,
                                           maxPeaks=100,
                                           sampleRate=defaultSampleRate,
                                           magnitudeThreshold=0,
                                           orderBy="magnitude")

        # convert unit to cents, PitchSalienceFunction takes 55 Hz as the
        # default reference
        run_pitch_salience_function = estd.PitchSalienceFunction()
        run_pitch_salience_function_peaks = estd.PitchSalienceFunctionPeaks()
        run_pitch_contours = estd.PitchContours(pitchContinuity=pc,timeContinuity=tc) # default params

        # compute frame by frame
        pool = Pool()
        for frame in estd.FrameGenerator(audio, frameSize=defaultFrameSize,
                                         hopSize=defaultHopSize):
            frame = run_windowing(frame)
            spectrum = run_spectrum(frame)
            peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
            salience = run_pitch_salience_function(peak_frequencies,
                                                   peak_magnitudes)
            salience_peaks_bins, salience_peaks_contour_saliences = \
                run_pitch_salience_function_peaks(salience)
            if not np.size(salience_peaks_bins):
                salience_peaks_bins = np.array([0])
            if not np.size(salience_peaks_contour_saliences):
                salience_peaks_contour_saliences = np.array([0])

            pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
            pool.add('allframes_salience_peaks_contourSaliences',
                     salience_peaks_contour_saliences)

        # post-processing: contour tracking
        contours_bins, contour_saliences, contours_start_times, duration = \
            run_pitch_contours(
                pool['allframes_salience_peaks_bins'],
                pool['allframes_salience_peaks_contourSaliences'])
        return contours_bins, contours_start_times, contour_saliences, duration

    @staticmethod
    def _join_contours(pitch_contours_no_overlap, contour_saliences_no_overlap,
                       start_samples_no_overlap, num_samples):
        # accumulate pitch and salience
        pitch = np.array([0.] * num_samples)
        salience = np.array([0.] * num_samples)
        for i in range(0, len(pitch_contours_no_overlap)):
            start_samp = start_samples_no_overlap[i]
            end_samp = start_samples_no_overlap[i] + len(
                pitch_contours_no_overlap[i])

            try:
                pitch[start_samp:end_samp] = pitch_contours_no_overlap[i]
                salience[start_samp:end_samp] = contour_saliences_no_overlap[i]
            except ValueError:
                warnings.warn("The last pitch contour exceeds the audio "
                              "length. Trimming...")

                pitch[start_samp:] = pitch_contours_no_overlap[i][:len(
                    pitch) - start_samp]
                salience[start_samp:] = contour_saliences_no_overlap[i][:len(
                    salience) - start_samp]
        return pitch, salience

    @staticmethod
    def _remove_overlaps(start_samples, pitch_contours, contour_saliences,lens, acc_idx):
        # remove overlaps
        rmv_idx = []
        for i in range(0, len(start_samples)):
            # print '_' + str(i)
            # create the sample index vector for the checked pitch contour
            curr_samp_idx = range(start_samples[i], start_samples[i] + lens[i])

            # get the non-overlapping samples
            curr_samp_idx_no_overlap = list(set(curr_samp_idx) -
                                            set(acc_idx))
            if curr_samp_idx_no_overlap:
                temp = min(curr_samp_idx_no_overlap)
                keep_idx = range(temp - start_samples[i],
                                 (max(curr_samp_idx_no_overlap) -
                                  start_samples[i]) + 1)

                # remove all overlapping values
                pitch_contours[i] = np.array(pitch_contours[i])[keep_idx]
                contour_saliences[i] = np.array(contour_saliences[i])[keep_idx]
                # update the startSample
                start_samples[i] = temp
            else:  # totally overlapping
                rmv_idx.append(i)

        # remove totally overlapping pitch contours
        rmv_idx = sorted(rmv_idx, reverse=True)
        for r in rmv_idx:
            pitch_contours.pop(r)
            contour_saliences.pop(r)
            start_samples.pop(r)

        return start_samples, pitch_contours, contour_saliences

suite = allTests(TestPitchContours)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
