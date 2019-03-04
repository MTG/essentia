#!/usr/bin/env python

# Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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

class TestPitchYinProbabilistic(TestCase):

    def testEmpty(self):
        self.assertComputeFails(PitchYinProbabilistic(), [])

    def testZero(self):
        pitch, voicedProbs = PitchYinProbabilistic()(zeros(2048))
        self.assertEqual(pitch, -61.735)
        self.assertEqual(voicedProbs, 0)

    def testSine(self):
        sr = 44100
        size = sr*1;
        freq = 440
        signal = [sin(2.0*pi*freq*i/sr) for i in range(size)]
        self.runTest(signal, sr, freq)

    def testBandLimitedSquare(self):
        sr = 44100
        size = sr*1;
        freq = 660
        w = 2.0*pi*freq
        nharms = 10
        signal = zeros(size)
        for i in range(size):
            for harm in range(nharms):
                signal[i] += .5/(2.*harm+1)*sin((2*harm+1)*i*w/sr)

        self.runTest(signal, sr, freq)

    def testBandLimitedSaw(self):
        sr = 44100
        size = sr*1;
        freq = 660
        w = 2.0*pi*freq
        nharms = 10
        signal = zeros(size)
        for i in range(1,size):
            for harm in range(1,nharms+1):
                signal[i] += 1./harm*sin(harm*i*w/sr)
        self.runTest(signal, sr, freq, 1.1, 0.1)

    def testBandLimitedSawMasked(self):
        sr = 44100
        size = sr*1;
        freq = 440
        w = 2.0*pi*freq
        subw = 2.0*pi*(freq-100)
        nharms = 10
        signal = zeros(size)
        for i in range(1,size):
            # masking noise:
            whitenoise = 2*(random.rand(1)-0.5)
            signal[i] += 2*whitenoise
            for harm in range(1,nharms):
                signal[i] += 1./harm*sin(i*harm*w/sr)
        signal = 5*LowPass()(signal)
        for i in range(1,size):
            for harm in range(1,nharms+1):
                signal[i] += .1/harm*sin(i*harm*w/sr)
            signal[i] += 0.5*sin(i*subw/sr)
        max_signal = max(signal) + 1
        signal = signal/max_signal
        self.runTest(signal, sr, freq, 1.5, 0.3)

    def runTest(self, signal, sr, freq, pitch_precision = 1, vp_precision = 0.1):
        frameSize = 1024
        hopsize = frameSize

        frames = FrameGenerator(signal, frameSize=frameSize, hopSize=hopsize)
        pitchDetect = PitchYinProbabilistic(frameSize=frameSize, sampleRate = sr)
        pitch = []
        voicedProbs = []
        for frame in frames:
            f, vp = pitchDetect(frame)
            pitch += [f]
            voicedProbs += [vp]
        self.assertAlmostEqual(mean(f), freq, pitch_precision)

    def testInvalidParam(self):
        self.assertConfigureFails(PitchYinProbabilistic(), {'frameSize' : 1})
        self.assertConfigureFails(PitchYinProbabilistic(), {'sampleRate' : 0})

    def testARealCase(self):
        frameSize = 2048
        sr = 48000
        hopSize = 256
        filename = join(testdata.audio_dir, 'recorded', 'long_voice.flac')
        audio = MonoLoader(filename=filename, sampleRate=sr)()
        frames = FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)
        pitchDetect = PitchYinProbabilistic(frameSize=frameSize, sampleRate = sr, hopSize=hopSize, outputUnvoiced="zero")
        pitch, _ = pitchDetect(audio)
        expected_pitch = readVector(join(filedir(), 'pitchyin/long_voice.txt'))
        self.assertAlmostEqualVector(pitch, expected_pitch)

suite = allTests(TestPitchYinProbabilistic)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
