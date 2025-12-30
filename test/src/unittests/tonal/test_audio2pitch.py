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
from numpy import sin, pi, mean, random


class TestAudio2Pitch(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Audio2Pitch(), [])

    def testZero(self):
        pitch, confidence, loudness, voiced = Audio2Pitch()(zeros(1024))
        self.assertEqual(pitch, 0)
        self.assertEqual(confidence, 0)
        self.assertEqual(loudness, 0.0)
        self.assertEqual(voiced, 0)

    def testSine(self):
        sample_rate = 44100
        size = sample_rate * 1
        frequency = 440
        amplitude_in_db = -3
        signal = [sin(2.0 * pi * frequency * i / sample_rate) for i in range(size)]
        self.runTest(
            signal, sample_rate, amplitude_in_db, frequency, loudness_precision=0.5
        )

    def testBandLimitedSquare(self):
        sample_rate = 44100
        size = sample_rate * 1
        frequency = 660
        w = 2.0 * pi * frequency
        nharms = 10
        amplitude = 0.5
        amplitude_in_db = -9
        signal = zeros(size)
        for i in range(size):
            for harm in range(nharms):
                signal[i] += (
                    amplitude
                    / (2.0 * harm + 1)
                    * sin((2 * harm + 1) * i * w / sample_rate)
                )

        self.runTest(signal, sample_rate, amplitude_in_db, frequency)

    def testBandLimitedSaw(self):
        sample_rate = 44100
        size = sample_rate * 1
        frequency = 660
        w = 2.0 * pi * frequency
        nharms = 10
        amplitude = 1.0
        amplitude_in_db = -1.43
        signal = zeros(size)
        for i in range(1, size):
            for harm in range(1, nharms + 1):
                signal[i] += amplitude / harm * sin(harm * i * w / sample_rate)
        self.runTest(
            signal,
            sample_rate,
            amplitude_in_db,
            frequency,
            pitch_precision=1.1,
            loudness_precision=0.2,
        )

    def testBandLimitedSawMasked(self):
        sample_rate = 44100
        size = sample_rate * 1
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
                signal[i] += 1.0 / harm * sin(i * harm * w / sample_rate)
        signal = 5 * LowPass()(signal)
        for i in range(1, size):
            for harm in range(1, nharms + 1):
                signal[i] += 0.1 / harm * sin(i * harm * w / sample_rate)
            signal[i] += 0.5 * sin(i * subw / sample_rate)
        max_signal = max(signal) + 1
        signal = signal / max_signal
        amplitude_in_db = -9
        self.runTest(
            signal,
            sample_rate,
            amplitude_in_db,
            freq,
            pitch_precision=1.5,
            conf_precision=0.3,
        )

    def runTest(
        self,
        signal: numpy.ndarray,
        sample_rate: int,
        amplitude_in_db: float,
        frequency: float,
        pitch_precision: float = 1,
        conf_precision: float = 0.1,
        loudness_precision: float = 0.1,
    ):
        frameSize = 1024
        hopsize = frameSize

        frames = FrameGenerator(signal, frameSize=frameSize, hopSize=hopsize)
        pitchDetect = Audio2Pitch(frameSize=frameSize, sampleRate=sample_rate)
        n_outputs = len(pitchDetect.outputNames())
        pitch, confidence, loudness, voiced = ([] for _ in range(n_outputs))
        for frame in frames:
            f, conf, l, v = pitchDetect(frame)
            pitch += [f]
            confidence += [conf]
            loudness += [amp2db(l)]
            voiced += [v]
        self.assertAlmostEqual(mean(f), frequency, pitch_precision)
        self.assertAlmostEqual(mean(confidence), 1, conf_precision)
        self.assertAlmostEqual(mean(loudness), amplitude_in_db, loudness_precision)
        self.assertAlmostEqual(mean(voiced), 1, conf_precision)

    def testInvalidParam(self):
        self.assertConfigureFails(Audio2Pitch(), {"frameSize": 1})
        self.assertConfigureFails(Audio2Pitch(), {"sampleRate": 0})
        self.assertConfigureFails(
            Audio2Pitch(), {"sampleRate": 44100, "maxFrequency": 44100}
        )
        self.assertConfigureFails(
            Audio2Pitch(),
            {"sampleRate": 44100, "maxFrequency": 200, "minFrequency": 250},
        )
        self.assertConfigureFails(
            Audio2Pitch(),
            {"sampleRate": 44100, "pitchAlgorithm": "yin_fft"},
        )
        self.assertConfigureFails(
            Audio2Pitch(),
            {"sampleRate": 44100, "loudnessThreshold": 1.0},
        )
        self.assertConfigureFails(
            Audio2Pitch(),
            {"sampleRate": 44100, "pitchConfidenceThreshold": -0.5},
        )
        self.assertConfigureFails(
            Audio2Pitch(),
            {"sampleRate": 44100, "pitchConfidenceThreshold": 1.5},
        )

    def testARealCase(self):
        # The expected values were recomputed from commit
        # 2d37c0713fb6cc5f637b3d8f5d65aa90b36d4277
        #
        # The expeted values were compared with the vamp pYIN
        # implementation of the YIN algorithm producing very
        # similar values.
        #
        # https://code.soundsoftware.ac.uk/projects/pyin

        frameSize = 1024
        sample_rate = 44100
        hopSize = 512
        loudness_threshold = -80
        filename = join(testdata.audio_dir, "recorded", "vignesh.wav")
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frames = FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)
        pitchDetect = Audio2Pitch(
            frameSize=frameSize,
            sampleRate=sample_rate,
            pitchConfidenceThreshold=0.15,
            loudnessThreshold=loudness_threshold,
        )

        n_outputs = len(pitchDetect.outputNames())
        pitch, confidence, loudness, voiced = ([] for _ in range(n_outputs))
        for frame in frames:
            f, conf, l, v = pitchDetect(frame)
            pitch += [f]
            confidence += [conf]
            loudness += [l]
            voiced += [v]
        expected_pitch = numpy.load(join(filedir(), "pitchyinfft/vignesh_pitch.npy"))
        expected_conf = numpy.load(
            join(filedir(), "pitchyinfft/vignesh_confidence.npy")
        )
        expected_voiced = [1] * len(expected_pitch)
        self.assertAlmostEqualVector(pitch, expected_pitch, 1e-6)
        self.assertAlmostEqualVector(confidence, expected_conf, 5e-5)
        self.assertAlmostEqualVector(voiced, expected_voiced)


suite = allTests(TestAudio2Pitch)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
