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
from essentia import Pool, array
from essentia.standard import MonoLoader, PercivalEvaluatePulseTrains, HarmonicBpm, FrequencyBands, NoveltyCurve 
import numpy as np

class TestPercivalEvaluatePulseTrains(TestCase):

    def testEmpty(self):        
        lag = PercivalEvaluatePulseTrains()([],[])
        self.assertEqual(-1.0, lag)

    def testZero(self):
        zeroOSS = zeros(10000)
        zeropositions = zeros(10000)
        lag = PercivalEvaluatePulseTrains()(zeroOSS, zeropositions)
        self.assertEqual(0.0, lag)

    def testConstantInput(self):
        onesOSS = ones(10000)
        onespositions = ones(10000)
        lag = PercivalEvaluatePulseTrains()(onesOSS, onespositions)              
        self.assertEqual(1.0, lag)
    
    # FIXME- A better comparison model is required for this regression test.
    # onsets and position inuts calculated using essentia functions.
    # Alternative sources should be used.
    # For original paper refer to 
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6879451
    # https://github.com/marsyas/marsyas
    def testRegression(self, percivalevaluatepulsetrains=None):
        fs = 44100
        durInSecs = 3
        inputSize=44100*durInSecs # Take 3 secs of samples

        if not percivalevaluatepulsetrains:
            percivalevaluatepulsetrains = PercivalEvaluatePulseTrains()

        # Calculates the positions and Peaks
        config = { 'range': inputSize -1, 'maxPosition': inputSize, 'minPosition': 0, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        audio = audio[:durInSecs * fs]#let's use only the first two seconds of the signals
        audio = audio / np.max(np.abs(audio))

        # Loading Essentia functions required
        # See Essenatia tutorial example: 
        # https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_onsetdetection.py
        odf = OnsetDetection(method = 'flux')
        w = Windowing(type = 'hann')
        fft = FFT() # this gives us a complex FFT
        spectrum= Spectrum()
        onsets = Onsets()

        
        #Essentia beat tracking
        pool = Pool()
        for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
            pool.add('features.flux', odf(spectrum(w(frame)), zeros(len(frame))))            

        onsets_flux = onsets(array([pool['features.flux']]),[1])
        onsets_flux = onsets_flux[onsets_flux<durInSecs]                    
        (posis, vals) = pdetect(audio)

        # oss (vector_real) - onset strength signal (or other novelty curve)
        # positions (vector_real) - peak positions of BPM candidates
        lag = PercivalEvaluatePulseTrains()(onsets_flux, posis) 
            
        # Previously measured value for lag was 107761
        self.assertEqual(107761.0, lag)        
  
    # FIXME: Failed Test Case. what param do we send to reset(...)?
    def testResetMethod(self):
        percivalevaluatepulsetrains = PercivalEvaluatePulseTrains()
        
        self.testRegression()
        percivalevaluatepulsetrains.reset()
        self.testRegression()

suite = allTests(TestPercivalEvaluatePulseTrains)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
