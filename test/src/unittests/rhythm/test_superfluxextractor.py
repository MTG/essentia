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
from essentia.standard import MonoLoader, SuperFluxExtractor
from numpy import *

class TestSuperFluxExtractor(TestCase):


    def testInvalidParam(self):
        # All the parameters ranges are theoretically from 0 to infinity
        # Hence, we use neg. value for invalid checks.
        self.assertConfigureFails(SuperFluxExtractor(), { 'combine': -1})
        self.assertConfigureFails(SuperFluxExtractor(), { 'frameSize': -1})
        self.assertConfigureFails(SuperFluxExtractor(), { 'hopSize': -1})
        self.assertConfigureFails(SuperFluxExtractor(), { 'ratioThreshold': -1})
        self.assertConfigureFails(SuperFluxExtractor(), { 'sampleRate': -1})
        self.assertConfigureFails(SuperFluxExtractor(), { 'threshold': -1})                  

    def testRegressionDubstep(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'dubstep.wav'))()
      
        # This test case will use the documented default parameters from recording dubstep.wav
        onsets = SuperFluxExtractor(combine=30,frameSize=2048,hopSize=256,ratioThreshold=16,
            sampleRate=44100,threshold=0.5)(audio)
        
        # This commented out code was used to obtain reference samples for storing in a file.
        # save('superfluxdub', onsets)        
        
        # Reference samples are loaded as expected values
        expected_superflux = load(join(filedir(), 'superflux/superfluxdub.npy'))
        self.assertAlmostEqualVector(onsets, expected_superflux, 1e-5)

    def testRegressionTechnoloop(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
       
        # This test case will use peak parameters slighlt ifferent from default from recording techno_loop.wav
        onsets = SuperFluxExtractor(combine=20,frameSize=2048,hopSize=256,ratioThreshold=8,
            sampleRate=44100,threshold=0.25)(audio)
        
        # This commented out code was used to obtain reference samples for storing in a file.
        # save('superfluxtechno', onsets)        
        
        # Reference samples are loaded as expected values
        expected_superflux = load(join(filedir(), 'superflux/superfluxtechno.npy'))
        self.assertAlmostEqualVector(onsets, expected_superflux, 1e-5)

    def _assertVectorWithinVector(self, found, expected, precision=1e-5):
        for i in range(len(found)):
            for j in range(1,len(expected)):
                if found[i] <= expected[j] and found[i] >= expected[j-1]:
                    if fabs(found[i] - expected[j-1]) < fabs(expected[j] - found[i]):
                        self.assertAlmostEqual(found[i], expected[j-1], precision)
                    else:
                        self.assertAlmostEqual(found[i], expected[j], precision)

    def testSilence(self):
        # zeros should return no onsets (empty array)
        self.assertEqualVector(SuperFluxExtractor()(zeros(44100)), [])

    def testEmpty(self):
        # empty input should return no onsets (empty array)
        self.assertEqualVector(SuperFluxExtractor()([]), [])

    def testImpulse(self):
        # Given an impulse should return its position
        sampleRate = 44100
        frameSize = 2048
        hopSize = 256
        signal = zeros(sampleRate * 2)
        # impulses at 0:30 and 1:00
        signal[22050] = 1.
        signal[44100] = 1.

        expected = [0.5, 1.]

        result = SuperFluxExtractor(sampleRate=sampleRate, frameSize=frameSize,
                                    hopSize=hopSize)(signal)

        # SuperfluxPeaks has a parameter 'combine' which is a threshold that
        # puts together consecutive peaks. This means that a peak will be
        # detected as soon as it is seen by a frame. Thus, the frame size
        # also influences the expected precision of the algorithm.
        precision = (hopSize + frameSize) / sampleRate
        self.assertAlmostEqualVectorAbs(result, expected, precision)

suite = allTests(TestSuperFluxExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
