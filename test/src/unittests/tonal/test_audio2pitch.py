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
from numpy import sin, pi, mean, random, sqrt


class TestAudio2Pitch(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Audio2Pitch(), [])

    def testZero(self):
        pitch, confidence, voiced, loudness = Audio2Pitch()(zeros(1024))
        self.assertEqual(pitch, 0)
        self.assertEqual(confidence, 0)
        self.assertEqual(voiced, 0)
        self.assertEqual(loudness, 0)

    def testSine(self):
        sr = 44100
        size = sr * 1
        freq = 440
        signal = [sin(2.0 * pi * freq * i / sr) for i in range(size)]
        self.runTest(signal, sr, 1, freq)

    def testBandLimitedSquare(self):
        sr = 44100
        size = sr * 1
        freq = 660
        w = 2.0 * pi * freq
        nharms = 10
        amplitude = 0.5
        signal = zeros(size)
        for i in range(size):
            for harm in range(nharms):
                signal[i] += (
                    amplitude / (2.0 * harm + 1) * sin((2 * harm + 1) * i * w / sr)
                )

        self.runTest(signal, sr, amplitude, freq)

    def runTest(
        self, signal, sr, amplitude, freq, pitch_precision=1, conf_precision=0.1
    ):
        frameSize = 1024
        hopsize = frameSize

        frames = FrameGenerator(signal, frameSize=frameSize, hopSize=hopsize)
        pitchDetect = Audio2Pitch(frameSize=frameSize, sampleRate=sr)
        pitch, confidence, loudness, voiced = ([] for _ in range(4))
        for frame in frames:
            f, conf, l, v = pitchDetect(frame)
            pitch += [f]
            confidence += [conf]
            loudness += [l]
            voiced += [v]
        self.assertAlmostEqual(mean(f), freq, pitch_precision)
        self.assertAlmostEqual(mean(confidence), 1, conf_precision)
        self.assertAlmostEqual(mean(loudness), amplitude / sqrt(2), conf_precision)
        self.assertAlmostEqual(mean(voiced), 1, conf_precision)


suite = allTests(TestAudio2Pitch)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
