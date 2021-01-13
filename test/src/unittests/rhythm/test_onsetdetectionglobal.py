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
from essentia.standard import MonoLoader, OnsetDetectionGlobal as stdOnsetDetectionGlobal

framesize = 1024
hopsize = 512

class TestOnsetDetectionGlobal(TestCase):

    def testZero(self):
        # Inputting zeros should return no onsets(empty array)
        audio=zeros(44100*5)
        onsetdetectionglobal = stdOnsetDetectionGlobal()
        onset_beat_emphasis=OnsetDetectionGlobal(method='beat_emphasis')
        onset_infogain=OnsetDetectionGlobal(method='infogain')
        self.assertEqualVector(onset_beat_emphasis(audio),zeros(len(onset_beat_emphasis(audio))))
        self.assertEqualVector(onset_infogain(audio),zeros(len(onset_infogain(audio))))
     
    def testInvalidParam(self):
        self.assertConfigureFails(OnsetDetectionGlobal(),{'sampleRate':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(),{'method':'unknown'})
        self.assertConfigureFails(OnsetDetectionGlobal(),{'hopSize':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(),{'frameSize':-1})

    def testRegressionTest(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
        
        # This test was developed specifically with observed values from the "techno_loop.wav" file 
        # 1. Print the first 3 and last 3 elements of beat_emphasis save these values in a beat_emphasis_list
        # 2. Print the first 3 and last 3 elements of infogain like in 1.
        # 3. These printed elements are the expected values 
        # 4. Use these expected values for comparison in a self.assertAlmostEqual()
        
        onsetdetectionglobal_infogain = stdOnsetDetectionGlobal(method='infogain')
        onsetdetectionglobal_beat_emphasis = stdOnsetDetectionGlobal(method = 'beat_emphasis')
        beat_emphasis_list = onsetdetectionglobal_beat_emphasis(audio).tolist()
        infogain_list = onsetdetectionglobal_infogain(audio).tolist()

        
        self.assertAlmostEqual(beat_emphasis_list[0], 5.7949586)
        self.assertAlmostEqual(beat_emphasis_list[1], 1.1640115e+01 )
        self.assertAlmostEqual(beat_emphasis_list[2],8.4264336e+00)
        self.assertAlmostEqual(beat_emphasis_list[2460],31.080015182495117)
        self.assertAlmostEqual(beat_emphasis_list[2461],18.669902801513672)
        self.assertAlmostEqual(beat_emphasis_list[2462],6.2911224365234375)
        self.assertAlmostEqual(infogain_list[0],0)
        self.assertAlmostEqual(infogain_list[1],0)
        self.assertAlmostEqual(infogain_list[2],0)
        self.assertAlmostEqual(infogain_list[2640],91.23327)
        self.assertAlmostEqual(infogain_list[2641],74.447784)
        self.assertAlmostEqual(infogain_list[2642],64.83572)

    # This unit test is s simplified verion of Regression Test but using default method
    # and different array points.
    # The default method of OnsetDEtectionGlobal is infogain
    def testRegressionTestDefaultMethod(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        onsetdetectionglobal = stdOnsetDetectionGlobal()
        onsetDetections = onsetdetectionglobal(audio)
        self.assertAlmostEqual(onsetDetections[2640],91.23327,0.01)
        self.assertAlmostEqual(onsetDetections[2641],74.447784,0.01)


suite=allTests(TestOnsetDetectionGlobal)

if __name__=='__main__':
    TextTestRunner(verbosity=2).run(suite)
