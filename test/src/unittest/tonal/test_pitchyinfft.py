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
from numpy import sin, pi, mean, random

class TestPitchYinFFT(TestCase):

    def testEmpty(self):
        self.assertComputeFails(PitchYinFFT(), [])

    def testZero(self):
        pitch, confidence = PitchYinFFT()(zeros(1024))
        self.assertEqual(pitch, 0)
        self.assertEqual(confidence, 0)


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


    def runTest(self, signal, sr, freq, pitch_precision = 1, conf_precision = 0.1):
        frameSize = 1024
        hopsize = frameSize

        frames = FrameGenerator(signal, frameSize=frameSize, hopSize=hopsize)
        win = Windowing(type='hann')
        pitchDetect = PitchYinFFT(frameSize=frameSize, sampleRate = sr)
        pitch = []
        confidence = []
        for frame in frames:
            spec = Spectrum()(win(frame))
            f, conf = pitchDetect(spec)
            pitch += [f]
            confidence += [conf]
        self.assertAlmostEqual(mean(f), freq, pitch_precision)
        self.assertAlmostEqual(mean(confidence), 1, conf_precision)

    def testInvalidParam(self):
        self.assertConfigureFails(PitchYinFFT(), {'frameSize' : 1})
        self.assertConfigureFails(PitchYinFFT(), {'sampleRate' : 0})

    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded','mozart_c_major_30sec.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frames = FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)
        win = Windowing(type='hann')
        pitchDetect = PitchYinFFT(frameSize=frameSize, sampleRate = sr)
        pitch = []
        confidence = []
        for frame in frames:
            spec = Spectrum()(win(frame))
            f, conf = pitchDetect(spec)
            pitch += [f]
            confidence += [conf]
        expected_pitch = readVector(join(filedir(), 'pitchyinfft/pitch_mozart_c_major_30sec.txt'))
        expected_conf = readVector(join(filedir(), 'pitchyinfft/pitchconfidence_mozart_c_major_30sec.txt'))
        self.assertAlmostEqualVector(pitch, expected_pitch)
        self.assertAlmostEqualVector(confidence, expected_conf, 5e-5)



suite = allTests(TestPitchYinFFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
