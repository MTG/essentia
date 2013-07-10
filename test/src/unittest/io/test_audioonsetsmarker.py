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
from numpy import pi, sin
import os

import essentia.streaming as es

file = "foo.wav"

class TestAudioOnsetsMarker(TestCase):

    def testRegression(self):
        sr = 44100
        inputSize = sr # 1 second of audio
        input = [0.5*sin(2.0*pi*440.0*i/inputSize) for i in range(inputSize)]

        onsets = [0.15, 0.5, 0.9]
        audio = AudioOnsetsMarker(sampleRate = sr, onsets=onsets)(input)
        MonoWriter(filename=file)(audio)
        left = MonoLoader(filename = file, downmix='left', sampleRate=sr)()
        diff = zeros(inputSize)
        for i in range(inputSize):
          # due to AudioOnsetsMarker downmixing by 0.5,
          #input[i] must be divided accordingly:
          diff[i] = left[i] - input[i]/2

        os.remove(file)
        onsetWidth = 0.04*sr
        epsilon = 1e-3
        found = zeros(len(onsets))
        j = 0
        i = 0
        while i < inputSize:
          if diff[i] > epsilon:
            found[j] = float(i)/float(sr)
            j += 1
            i += onsetWidth
          else: i+=1
        self.assertAlmostEqualVector(found, onsets, 1e-3)

    def testEmpty(self):
        audio = AudioOnsetsMarker(sampleRate = 44100, onsets=[1, 2, 3])([])
        MonoWriter(filename=file)(audio)
        self.assertTrue( not os.path.exists(file) )

    def testEmptyOnsets(self):
        signal = [0.1, 0.2, 0.3]
        audio = AudioOnsetsMarker(sampleRate = 44100, onsets=[])(signal)
        MonoWriter(filename=file)(audio)
        mix = MonoLoader(filename = file, downmix='mix', sampleRate=44100)()
        os.remove(file)
        expected = [0.5*val for val in signal]
        self.assertAlmostEqualVector(mix, expected, 1e-3)

    def testStreaming(self):
        signal = [0]*44100
        onsets = [0, 0.05, 0.07, 0.08, 0.085,  0.1, 0.5, 0.8, 0.9]

        # standard version works correctly, and will be used as a reference:
        expected = AudioOnsetsMarker(sampleRate = 44100, onsets=onsets)(signal)

        # streaming version is the one that fails:
        pool = Pool()
        gen = es.VectorInput(signal)
        onsetMarker = es.AudioOnsetsMarker(sampleRate = 44100, onsets=onsets)

        gen.data >> onsetMarker.signal
        onsetMarker.signal >> (pool, "markedOnsets")
        run(gen)

        self.assertEqualVector(pool['markedOnsets'], expected)



    def testInvalidParam(self):
        self.assertConfigureFails(AudioOnsetsMarker(), { 'sampleRate' : 0 })
        self.assertConfigureFails(AudioOnsetsMarker(), { 'type' : 'burst' })

suite = allTests(TestAudioOnsetsMarker)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
