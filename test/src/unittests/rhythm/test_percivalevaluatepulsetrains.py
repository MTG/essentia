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
from essentia.standard import MonoLoader, PercivalEvaluatePulseTrains, HarmonicBpm, FrequencyBands, NoveltyCurve 

class TestPercivalEvaluatePulseTrains(TestCase):

    def testEmpty(self):        
        lag = PercivalEvaluatePulseTrains()([],[])
        self.assertEqual(-1.0, lag)

    def testZero(self):
        zeroOSS = zeros(10000)
        zeropositions = zeros(10000)
        lag = PercivalEvaluatePulseTrains()(zeroOSS, zeropositions)
        print(lag)
        self.assertEqual(0.0, lag)

    def testConstantInput(self):
        onesOSS = ones(10000)
        onespositions = ones(10000)
        lag = PercivalEvaluatePulseTrains()(onesOSS, onespositions)              
        self.assertEqual(1.0, lag)
    
    """
    FIXME
    def testRegression(self):
        # Calculates the positions and Peaks
        config = { 'range': inputSize -1, 'maxPosition': inputSize, 'minPosition': 0, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()

        (posis, vals) = pdetect(audio)

        # Calculates the OSS
        fc = FrameCutter(frameSize=4096, hopSize=512)
        windower = Windowing(type='blackmanharris62')
        specAlg = Spectrum(size=4096)
        fluxAlg = Flux()
        # Calculate the average flux over all frames of audio
        frame = fc(audio)
        fluxSum = 0
        count = 0
        while len(frame) != 0:
            spectrum = specAlg(windower(frame))
            fluxSum += fluxAlg(spectrum)

            count += 1
            frame = fc(audio)

        fluxAvg = float(fluxSum) / float(count)
        filteredSignal = LowPass(cutoffFrequency=1000)(fluxAvg)
           
        fc = FrameCutter(frameSize = len(audio),  hopSize = len(audio))
        oss = fc(filteredSignal)
        lag = PercivalEvaluatePulseTrains()(oss,posis() )
    """

    """
    reset test commented out for now. 
    TBD: what param do we send to reset(...)?
    def testResetMethod(self):
        self.testRegression()
        PercivalEvaluatePulseTrains.reset()
        self.testRegression()
    """

suite = allTests(TestPercivalEvaluatePulseTrains)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
