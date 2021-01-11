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
from essentia.standard import Spectrum as stdSpectrum

framesize = 1024
hopsize = 512

class TestOnsetDetectionGlobal(TestCase):

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        onsetdetectionglobal = stdOnsetDetectionGlobal()
        onsetDetections = onsetdetectionglobal(audio)
        self.assertAlmostEqual(onsetDetections[2640],91.23327,0.01)
        self.assertAlmostEqual(onsetDetections[2641],74.447784,0.01)

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
    

suite=allTests(TestOnsetDetectionGlobal)

if __name__=='__main__':
    TextTestRunner(verbosity=2).run(suite)
