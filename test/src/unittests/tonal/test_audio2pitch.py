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
        frequency = 440
        signal = [sin(2.0 * pi * frequency * i / sr) for i in range(size)]
        self.runTest(signal, sr, 1, frequency)

    def testBandLimitedSquare(self):
        sample_rate = 44100
        size = sample_rate * 1
        frequency = 660
        w = 2.0 * pi * frequency
        nharms = 10
        amplitude = 0.5
        signal = zeros(size)
        for i in range(size):
            for harm in range(nharms):
                signal[i] += (
                    amplitude
                    / (2.0 * harm + 1)
                    * sin((2 * harm + 1) * i * w / sample_rate)
                )

        self.runTest(signal, sample_rate, amplitude, frequency)

    def testBandLimitedSaw(self):
        sample_rate = 44100
        size = sample_rate * 1
        frequency = 660
        w = 2.0 * pi * frequency
        nharms = 10
        amplitude = 1.0
        signal = zeros(size)
        for i in range(1, size):
            for harm in range(1, nharms + 1):
                signal[i] += amplitude / harm * sin(harm * i * w / sample_rate)
        self.runTest(signal, sample_rate, 1.2, frequency, 1.1, 0.1)

    def testBandLimitedSawMasked(self):
        sr = 44100
        size = sr * 1
        freq = 440
        w = 2.0 * pi * freq
        subw = 2.0 * pi * (freq - 100)
        nharms = 10
        signal = zeros(size)
        for i in range(1, size):
            # masking noise:
            whitenoise = 2 * (random.rand(1) - 0.5)
            signal[i] += 2 * whitenoise
            for harm in range(1, nharms):
                signal[i] += 1.0 / harm * sin(i * harm * w / sr)
        signal = 5 * LowPass()(signal)
        for i in range(1, size):
            for harm in range(1, nharms + 1):
                signal[i] += 0.1 / harm * sin(i * harm * w / sr)
            signal[i] += 0.5 * sin(i * subw / sr)
        max_signal = max(signal) + 1
        signal = signal / max_signal
        self.runTest(signal, sr, 0.5, freq, 1.5, 0.3)

    def runTest(
        self,
        signal: numpy.ndarray,
        sample_rate: int,
        amplitude: float,
        frequency: float,
        pitch_precision: float = 1,
        conf_precision: float = 0.1,
    ):
        frameSize = 1024
        hopsize = frameSize

        frames = FrameGenerator(signal, frameSize=frameSize, hopSize=hopsize)
        pitchDetect = Audio2Pitch(frameSize=frameSize, sampleRate=sample_rate)
        pitch, confidence, loudness, voiced = ([] for _ in range(4))
        for frame in frames:
            f, conf, l, v = pitchDetect(frame)
            pitch += [f]
            confidence += [conf]
            loudness += [l]
            voiced += [v]
        self.assertAlmostEqual(mean(f), frequency, pitch_precision)
        self.assertAlmostEqual(mean(confidence), 1, conf_precision)
        self.assertAlmostEqual(mean(loudness), amplitude / sqrt(2), conf_precision)
        self.assertAlmostEqual(mean(voiced), 1, conf_precision)


suite = allTests(TestAudio2Pitch)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
