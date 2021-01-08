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

from essentia_test import *
import math

class TestTempoTapDegara(TestCase):

    def testInvalidParam(self):
    # testing that output is valid (not NaN nor inf nor negative time values)
    #but not testing for correct result as no ground truth is supplied
        self.assertConfigureFails(TempoTapDegara(), { 'maxTemp': 0})
        self.assertConfigureFails(TempoTapDegara(), { 'minTempo': 0 })
        self.assertConfigureFails(TempoTapDegara(), { 'resample': 0})
        self.assertConfigureFails(TempoTapDegara(), { 'sampleRateODF': -1})
    
    def testRegression(self, tempotapdegara  = None):
        # to match with test_tempotap.py
        numberFrames = 1024
        hopSize = 256

        if not tempotapdegara :
            tempotapdegara  = TempoTapDegara(frameHop = numberFrames,
                                          hopSize = hopSize)
        ticks = []
        matching_onsetDetections  = []
        for i in range(numberFrames*5):
            onsetDetections  = []
            if (i % numberFrames == numberFrames-1 ):
                # read output from tempotap, which will be produced after numberFrames
                # frames:
                onsetDetections = readMatrix(join(filedir(), 'tempotap', 'output.txt'))

            #ticks, matching_onsetDetections  = tempotapdegara (onsetDetections , phases)
            tmp = tempotapdegara (onsetDetections)
            ticks +=list(tmp[0])
            matching_onsetDetections  += list(tmp[1])

        self.assertEquals( True, len(ticks) > 0)
        # make sure that time is not negative
        self.assert_(all(array(ticks)) >= 0 )
        # make sure there is at least one matching period
        self.assertEquals( True, len(matching_onsetDetections ) > 0)


    def testResetMethod(self):
        # NB: this test should actually fail when not calling reset, which it
        # doesn't at the moment...
        numberFrames = 1024
        hopSize = 256
        tempotapdegara  = TempoTapDegara(frameHop = numberFrames,
                                      hopSize = hopSize)

        self.testRegression()
        tempotapdegara .reset()
        self.testRegression()


    def testImpulseTrain(self):
    # most of the code is borrowed from test_tempotap.py. Applying tempotaptics
    # to tempotap's output should yield tempotap's input (more or less)
        inputLength = 14200
        numberFrames = 1024
        featuresNumber = 1
        guessPeriod = 70
        guessPhase = 17
        hopSize = 256

        # tests that for an impulse will yield the correct position
        audiosize = 10000
        audio = zeros(audiosize)
        pos = 5.5  # impulse will be in between frames 4 and 5
        audio[int(floor(pos*(hopsize)))] = 1.
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize,
                startFromZero=True)
        win = Windowing(type='hamming', zeroPadding=framesize)
        fft = FFT()

        nframe = 0

        for frame in frames:
            mag, ph = CartesianTpoPolar()(fft(win(frame)))
            onsetDetections = onsetDetection(spectrum = mag, phase = ph)
            tempotapdegara = tempoTapDegara(onsetDetection = onsetDetections)
            nframe+=1
            self.assertEqualVector(temptapdegara,zeros(len(tempotapdegara)))




    def testImpulseTrainTaps(self):
        inputLength = 14200
        numberFrames = 1024
        featuresNumber = 1
        guessPeriod = 70
        guessPhase = 17

        # create an impulse train of phase 17 and period 70
        features = zeros((inputLength, featuresNumber))
        i = int(math.floor(guessPhase))
        while i < len(features):
            features[i][0] = 1
            i += int(math.floor(guessPhase))
        
        
        onsetDetections = onsetDetection(spectrum = periodEstimates,
                            phase = phaseEstimates)
        tempotapdegaraticks  = TempoTapDegara(onsetDetection= onsetDetections)
        ticks = []

        for i in range(inputLength):
            mag,ph = tempotap(features[i])
            onsetDetections = onsetDetection(spectrum = mag, phase = ph)
            tmp = tempotapdegaraticks(onsetDetections)
            ticks += list(tmp)

        # flush current buffer with zeros
        for i in range(inputLength % numberFrames, numberFrames):
            mag,ph = tempotap(features[i])
            onsetDetections = onsetDetection(spectrum = mag, phase = ph)
            tmp = tempotapdegaraticks(onsetDetections)
            ticks += list(tmp)

        # some ticks get lost at the end, but we just want to check that the
        # ones we found are correct
        initial_guessed_phase = guessPhase*hopSize/44100.
        delta_guessed_phase = guessPeriod*hopSize/44100.
          
        for i in range(min(len(ticks), expected_nTicks)):
            self.assertAlmostEqual(ticks[i],initial_guessed_phase + delta_guessed_phase*i,1e-2 )
        for period in matching_periods:
            self.assertAlmostEqual(period, guessPeriod, 1e-2)

    def testZero(self):
        tempotapdegara  = TempoTapDegara()

        ticks = []
        matching_onsetDetections  = []

        for i in range(1300):
            tmp = tempotapdegara (zeros(12), zeros(12))

            ticks += list(tmp[0])
            matching_onsetDetections  += list(tmp[1])

        self.assert_(all(array(ticks) == 0))
        self.assert_(all(array(matching_onsetDetections ) == 0))

    def testEmpty(self):
        tempotapdegara  = TempoTapDegara()

        ticks = []
        matching_onsetDetections  = []
        onsetDetections  = []
        phases = []

        for i in range(1300):
            ticks  = tempotapdegara (onsetDetections)
            self.assertEqualVector(ticks, [])
            self.assertEqualVector(matching_onsetDetections , [])



suite = allTests(TestTempoTapDegara)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
