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
        # Expected values for onset is 0, with constant "ones" input.
        # Expected values for duration is 2.9692516, with constant "ones" input.
        # Expected values for MIDIpitch is -36., with constant "ones" input.        
        # FIXME there should be no such thing as a negative pitch value-.
        self.assertEqualVector(onset, [0])
        self.assertAlmostEqualVector(duration, [2.9692516],8)
        self.assertEqualVector(MIDIpitch, [-36.])

    def testEmpty(self):
        pitch = [] 
        signal = ones(1000) 
        onset, duration, MIDIpitch = PitchContourSegmentation()( pitch,signal)        
        self.assertEqualVector(onset, [])
        self.assertEqualVector(duration, [])
        self.assertEqualVector(MIDIpitch, [])
   
    def testARealCaseDefault(self):
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

    # Test for different non-default settings of PitchContourSegmentation
    def testARealCase1(self):        
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)
      
        onset, duration, MIDIpitch = PitchContourSegmentation(minDuration=0.5,hopSize=256)( pitch,audio)

        referenceOnsets  = [0.2031746, 1.1435828, 1.6428118, 2.1420407, 2.64127,   3.6455328, 4.4408164] 
        referenceDurations = [0.9287982,  0.48761904, 0.48761904, 0.48761904, 0.7778685,  0.5514739, 1.4744672 ]
        referenceMIDIpitch = [56.,59., 59., 58., 54., 54., 56.]
         
        self.assertAlmostEqualVector(onset, referenceOnsets, 8)
        self.assertAlmostEqualVector(duration, referenceDurations, 8)
        self.assertAlmostEqualVector(MIDIpitch, referenceMIDIpitch, 8)

        onset, duration, MIDIpitch = PitchContourSegmentation(pitchDistanceThreshold=30,rmsThreshold=-1)( pitch,audio)

        referenceOnsets  = [0.1015873,  0.2031746,  0.55727893, 0.6559637,  0.7546485,  0.85333335,
         0.95201814, 1.0507029,  1.1755102 , 1.29161 ,   1.3902948,  1.4889796,
         1.5876644,  1.8227664,  1.9272562,  2.2204082,  2.3626304,  2.4932427,
         2.8009071 ] 
        referenceDurations =  [0.09578231, 0.34829932, 0.09287982, 0.09287982, 0.09287982, 0.09287982,
         0.09287982, 0.11900227, 0.11029478, 0.09287982, 0.09287982, 0.09287982,
         0.12190476, 0.09868481, 0.17124717,0.13641724, 0.12480725, 0.3018594,
         0.15673469] 
        referenceMIDIpitch = [56., 56., 58., 60., 58., 60., 59., 58., 59.,57., 53., 53., 55., 55., 54., 56.,56., 56.,54.]     

        self.assertAlmostEqualVector(onset, referenceOnsets, 8)
        self.assertAlmostEqualVector(duration, referenceDurations, 8)
        self.assertAlmostEqualVector(MIDIpitch, referenceMIDIpitch, 8)
    

        onset, duration, MIDIpitch = PitchContourSegmentation(pitchDistanceThreshold=100,rmsThreshold=-3)( pitch,audio)

        referenceOnsets  = [0.1015873,  0.5834014,  0.68208617, 0.78077096, 0.88816327, 1.0071656,
         1.3322449, 1.4309297,  1.5354195,  1.8227664,  1.9853061,  2.2204082,
         2.829932  ] 
        referenceDurations =  [0.47600907, 0.09287982, 0.09287982,0.1015873,  0.11319728, 0.31927437,
         0.09287982, 0.09868481, 0.17414966, 0.15673469, 0.11319728, 0.6037188,
         0.12770975]
        referenceMIDIpitch = [56., 60., 59., 58., 60., 58., 55., 52., 55., 55., 53., 56., 53.]

        self.assertAlmostEqualVector(onset, referenceOnsets, 8)
        self.assertAlmostEqualVector(duration, referenceDurations, 8)
        self.assertAlmostEqualVector(MIDIpitch, referenceMIDIpitch, 8)


suite = allTests(TestPitchContourSegmentation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
