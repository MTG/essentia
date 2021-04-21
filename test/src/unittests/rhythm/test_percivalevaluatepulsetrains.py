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


from essentia_test import *
from essentia.standard import MonoLoader, PercivalEvaluatePulseTrains, HarmonicBpm, FrequencyBands, NoveltyCurve 

class TestPercivalEvaluatePulseTrains(TestCase):

    def testEmpty(self):        
        lag = PercivalEvaluatePulseTrains()([],[])
        self.assertEqual(-1.0, lag)

    def testZero(self):
        zeroOSS = zeros(1024)
        zeropositions = zeros(1024)
        lag = PercivalEvaluatePulseTrains()(zeroOSS, zeropositions)
        self.assertEqual(0.0, lag)

    def testConstantInput(self):
        onesOSS = ones(1024)
        onespositions = ones(1024)
        lag = PercivalEvaluatePulseTrains()(onesOSS, onespositions)              
        self.assertEqual(1.0, lag)
    

    def testRegression(self):
        # Regression tests on calculation of the positions and Peaks.
        inputSize=21168 # N.B The beat period is 21168 samples for 125 bpm @ 44.1k samp. rate
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()

        # Calculates the positions and Peaks
        pdetect = PeakDetection()

        # Calculates the OSS
        fc = FrameCutter(frameSize=inputSize, hopSize=inputSize)
        windower = Windowing(type='blackmanharris62')
        specAlg = Spectrum(size=4096)
        fluxAlg = Flux()

        # Calculate the average flux over all frames of audio
        frame = fc(audio)
        fluxArray = []
      
        for frame in FrameGenerator(audio, frameSize = inputSize, hopSize = inputSize):
            spectrum = specAlg(windower(frame))
            fluxArray.append(fluxAlg(spectrum))
            frame = fc(audio)
        filteredSignal = LowPass(cutoffFrequency=8000)(fluxArray)
        
        # Calculate PercivalEvaluatePulseTrains on fluxArray
        aSignal =  AutoCorrelation()(fluxArray)      
        pHarm= PercivalEnhanceHarmonics()(aSignal)
        oss, posis= pdetect(pHarm)      
        lag = PercivalEvaluatePulseTrains()(fluxArray,posis)
        # Based on previous observations with fluxArray output originating from techno_loop
        self.assertEqual(8.0, lag)

        # Calculate PercivalEvaluatePulseTrains on filtered fluxArray
        aSignal =  AutoCorrelation()(filteredSignal)      
        pHarm= PercivalEnhanceHarmonics()(aSignal)
        oss, posis= pdetect(pHarm)      
        lag = PercivalEvaluatePulseTrains()(filteredSignal,posis)
        # Based on previous observations with ffiltered luxArray output originating from techno_loop
        self.assertEqual(7.0, lag)

        # TODO Add a test with an artificial signal (combination of sines) so by using peak alignment measures
        # after running the autocorrelation function, we would be able to predict time lag.

suite = allTests(TestPercivalEvaluatePulseTrains)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
