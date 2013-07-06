#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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
from essentia import *
from essentia.streaming import AudioOnsetsMarker, MonoWriter
from numpy import pi, sin
import os

file = "foo.wav"

class TestAudioOnsetsMarker_Streaming(TestCase):

    def testRegression(self):
        sr = 44100
        inputSize = sr # 1 second of audio
        input = [0.5*sin(2.0*pi*440.0*i/inputSize) for i in range(inputSize)]
        onsets = [0.15, 0.5, 0.9]

        signal = VectorInput(input)
        marker = AudioOnsetsMarker(sampleRate = sr, onsets=onsets)
        writer = MonoWriter(filename=file)
        signal.data >> marker.signal;
        marker.signal>> writer.audio;

        run(signal);

        left = MonoLoader(filename = file, downmix='left', sampleRate=sr)()
        diff = zeros(inputSize)
        for i in range(inputSize):
          # due to AudioOnsetsMarker downmixing by 0.5,
          # input[i] must be divided accordingly:
          diff[i] = left[i] - input[i]/2

        os.remove(file)
        onsetWidth = 0.04*sr
        epsilon = 1e-3
        found = []
        j = 0
        i = 0
        while i < inputSize:
          if diff[i] > epsilon:
            found.append(float(i)/float(sr))
            j += 1
            i += onsetWidth
          else: i+=1
        self.assertAlmostEqualVector(found, onsets, 1.5e-3)

    def testEmptySignal(self):
        sr = 44100
        signal = VectorInput([])
        onsets = [0.15, 0.5, 0.9]
        marker = AudioOnsetsMarker(sampleRate = sr, onsets=onsets)
        writer = MonoWriter(filename=file)
        signal.data >> marker.signal
        marker.signal >> writer.audio
        run(signal)
        self.assertTrue( not os.path.exists(file) )

    def testInvalidParam(self):
        self.assertConfigureFails(AudioOnsetsMarker(), { 'sampleRate' : 0 })
        self.assertConfigureFails(AudioOnsetsMarker(), { 'type' : 'burst' })
        self.assertConfigureFails(AudioOnsetsMarker(), { 'onsets' : [-1, -2, 9]})
        self.assertConfigureFails(AudioOnsetsMarker(), { 'onsets' : [1, -2, 9]})
        self.assertConfigureFails(AudioOnsetsMarker(), { 'onsets' : [2, 0, 9]})


suite = allTests(TestAudioOnsetsMarker_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
