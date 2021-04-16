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


from numpy import *
from essentia_test import *

class TestPitchContourSegmentation(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(PitchContourSegmentation(), {'hopSize': -1})
        self.assertConfigureFails(PitchContourSegmentation(), {'minDuration': -1})
        self.assertConfigureFails(PitchContourSegmentation(), {'pitchDistanceThreshold': -1})
        self.assertConfigureFails(PitchContourSegmentation(), {'rmsThreshold': 1})
        self.assertConfigureFails(PitchContourSegmentation(), {'sampleRate': -1})        
        self.assertConfigureFails(PitchContourSegmentation(), {'tuningFrequency': -1})
        self.assertConfigureFails(PitchContourSegmentation(), {'tuningFrequency': 50000})        
    
    def testZero(self):
        pitch = zeros(1024) 
        signal = zeros(1024)              
        onset, duration, MIDIpitch = PitchContourSegmentation()( pitch,signal)        
        self.assertEqualVector(onset, [])
        self.assertEqualVector(duration, [])
        self.assertEqualVector(MIDIpitch, [])

    def testOnes(self):
        pitch = ones(1024) 
        signal = ones(1024)
        onset, duration, MIDIpitch = PitchContourSegmentation()( pitch,signal)        
        self.assertEqualVector(onset, [0])
        self.assertAlmostEqualVector(duration, [2.9692516],8)
        self.assertEqualVector(MIDIpitch, [-36.])

    def testEmpty(self):
        pitch = [] 
        signal = []
        onset, duration, MIDIpitch = PitchContourSegmentation()( pitch,signal)        
        self.assertEqualVector(onset, [])
        self.assertEqualVector(duration, [])
        self.assertEqualVector(MIDIpitch, [])
   
    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)
      
        onset, duration, MIDIpitch = PitchContourSegmentation()( pitch,audio)

        referenceOnsets  = [0.1015873,  0.5717914,  0.6704762,  0.769161,  0.86784583, 1.0013605,
            1.2161452,  1.31483,    1.4135147,  1.5238096,  1.8227664,  1.9272562,
            2.2204082,  2.3800454,  2.818322 ]

        referenceDurations= [0.4643991,  0.09287982, 0.09287982, 0.09287982, 0.12770975, 0.20897959,
            0.09287982, 0.09287982, 0.1044898,  0.18575963, 0.09868481, 0.17124717,
            0.1538322,  0.43247166, 0.13931973]

        referenceMIDIpitch = [56., 59., 60., 58., 60., 58., 59., 56., 52., 55., 55., 54., 56., 56., 54.]
        
        self.assertAlmostEqualVector(onset, referenceOnsets, 8)
        self.assertAlmostEqualVector(duration, referenceDurations, 8)
        self.assertAlmostEqualVector(MIDIpitch, referenceMIDIpitch, 8)

suite = allTests(TestPitchContourSegmentation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
