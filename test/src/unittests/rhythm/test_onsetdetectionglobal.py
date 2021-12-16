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
from essentia.standard import MonoLoader, OnsetDetectionGlobal as stdOnsetDetectionGlobal

framesize = 1024
hopsize = 512


class TestOnsetDetectionGlobal(TestCase):

    def testZero(self):
        # Inputting zeros should return no onsets(empty array)
        audio=zeros(44100*5)
        onset_beat_emphasis=OnsetDetectionGlobal(method='beat_emphasis')
        onset_infogain=OnsetDetectionGlobal(method='infogain')
        self.assertEqualVector(onset_beat_emphasis(audio), zeros(428))
        self.assertEqualVector(onset_infogain(audio), zeros(428))
     
    def testInvalidParam(self):
        self.assertConfigureFails(OnsetDetectionGlobal(), {'sampleRate':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(), {'method':'unknown'})
        self.assertConfigureFails(OnsetDetectionGlobal(), {'hopSize':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(), {'frameSize':-1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        
        onsetdetectionglobal_infogain = stdOnsetDetectionGlobal(method='infogain')
        onsetdetectionglobal_beat_emphasis = stdOnsetDetectionGlobal(method = 'beat_emphasis')
        calculated_beat_emphasis = onsetdetectionglobal_infogain(audio).tolist()
        calculated_infogain = onsetdetectionglobal_beat_emphasis(audio).tolist()

        """
        This code stores reference values in a file for later loading.
        save('input_infogain.npy', calculated_beat_emphasis)
        save('input_beat_emphasis.npy', calculated_infogain)             
        """
        
        # Reference samples are loaded as expected values
        onsetdetectionglobal_infogain = load(join(filedir(), 'onsetdetectionglobal/infogain.npy'))
        onsetdetectionglobal_beat_emphasis = load(join(filedir(), 'onsetdetectionglobal/beat_emphasis.npy'))
        expected_infogain = onsetdetectionglobal_infogain.tolist()
        expected_beat_emphasis = onsetdetectionglobal_beat_emphasis.tolist()

        self.assertAlmostEqualVectorFixedPrecision(calculated_beat_emphasis, expected_beat_emphasis,2)
        self.assertAlmostEqualVectorFixedPrecision(calculated_infogain, expected_infogain,2)


suite=allTests(TestOnsetDetectionGlobal)

if __name__=='__main__':
    TextTestRunner(verbosity=2).run(suite)
    
