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
        # Calculates the positions and Peaks
        
        inputSize=21168 # N.B The beat period is 21168 samples for 125 bpm @ 44.1k samp. rate
        config = { 'range': inputSize -1, 'maxPosition': inputSize, 'minPosition': 0, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()

        # Calculates the positions and Peaks")
        config = { 'range': inputSize -1, 'maxPosition': inputSize, 'minPosition': 0, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)

        # Calculates the OSS")
        fc = FrameCutter(frameSize=inputSize, hopSize=inputSize)
        windower = Windowing(type='blackmanharris62')
        specAlg = Spectrum(size=4096)
        fluxAlg = Flux()
        # Calculate the average flux over all frames of audio")
        frame = fc(audio)
        windowedSignal = windower(frame)
        outputSpectrum = specAlg(windowedSignal)
        fluxArray = []
        count = 0
        while len(frame) != 0:
            spectrum = specAlg(windower(frame))
            fluxArray.append(fluxAlg(spectrum))
            count += 1
            frame = fc(audio)
        filteredSignal = LowPass(cutoffFrequency=8000)(fluxArray)
        fc = FrameCutter(frameSize = len(audio),  hopSize = len(audio))
        cutsignal = fc(filteredSignal)
        aSignal =  AutoCorrelation()(cutsignal)      
        pHarm= PercivalEnhanceHarmonics()(aSignal)
        oss, posis= pdetect(pHarm)      
        lag = PercivalEvaluatePulseTrains()(cutsignal,posis)

suite = allTests(TestPercivalEvaluatePulseTrains)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
