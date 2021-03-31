#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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


class TestPitchMelodia(TestCase):

    def testZero(self):
        signal = zeros(256)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(PitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(PitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': 2})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(PitchMelodia(), {'minFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': 2.1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(PitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(PitchMelodia(), {'referenceFrequency': -1})             
        self.assertConfigureFails(PitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(PitchMelodia(), {'timeContinuity': -1})

    def testOnes(self):
        signal = ones(256)
        pitch, confidence = PitchMelodia()(signal)
        print(pitch)
        print(confidence)
        self.assertAlmostEqualVector(pitch, [0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0.])

    def testEmpty(self):
        pitch, confidence = PitchMelodia()([])
        print(pitch)
        print(confidence)
        #self.assertComputeFails(PitchMelodia()(emptyPeakBins, emptyPeakSaliences))
    
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)

        #This code stores reference values in a file for later loading.
        save('pitchmelodiapitch.npy', pitch)             
        save('pitchmelodiaconfidence.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch.npy'))
        expectedPitchMelodiaPitch = loadedPitchMelodiaPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitchMelodiaPitch, 2)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence.npy'))
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 2)


suite = allTests(TestPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
