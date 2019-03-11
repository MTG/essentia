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


from essentia_test import *


class TestLoudnessEBUR128(TestCase):

    def testRegression(self):
        # The test audio files for loudness are provided in EBU Tech 3341
        # https://tech.ebu.ch/docs/tech/tech3341.pdf

        # M, S, I = -20 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', '1kHz_sine_-20LUFS-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqualVector(m, essentia.array([-20.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-20.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -20., 0.1)

        # M, S, I = -26 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', '1kHz_sine_-26LUFS-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqualVector(m, essentia.array([-26.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-26.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -26., 0.1)

        # M, S, I = -40 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', '1kHz_sine_-40LUFS-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqualVector(m, essentia.array([-40.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-40.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -40., 0.1)

        # M, S, I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-1-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqualVector(m, essentia.array([-23.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-23.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -23., 0.1)

        # M, S, I = -33 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-2-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqualVector(m, essentia.array([-33.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-33.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -33., 0.1)

        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-3-16bit-v02.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(i, -23., 0.1)

        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-4-16bit-v02.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(i, -23., 0.1)


        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-5-16bit-v02.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(i, -23., 0.1)

        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-7_seq-3342-5-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(i, -23., 0.1)

        # Test audio files for dynamic range are provided in EBU Tech Doc 3342
        # https://tech.ebu.ch/docs/tech/tech3342.pdf

        # LRA = 10 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-1-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(r, 10., 1.)

        # LRA = 5 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-2-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(r, 5., 1.)

        # LRA = 20 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-3-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(r, 20., 1.)

        # LRA = 15 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-4-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate)(audio)
        self.assertAlmostEqual(r, 15., 1.)

    def testRegressionStartAtZero(self):
        # The test audio files for loudness are provided in EBU Tech 3341
        # https://tech.ebu.ch/docs/tech/tech3341.pdf

        # Test zero-centered. When startAtZero=True, the loudness measurement
        # windows are centered at zero position in time producing the fadein
        # and fadeout effect at the start and the end of the signal. Estimate
        # the number of the affected values from windows/hop size

        # Default hopSize = 0.1 s
        # Momentary loudness window size = 0.4 s
        # Short-term loudness window size = 3 s

        fade_size_m = round((0.4 / 2) / 0.1)
        fade_size_s = round((3. / 2) / 0.1)

        # M, S, I = -20 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', '1kHz_sine_-20LUFS-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()

        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        m = m[fade_size_m:-fade_size_m]
        s = s[fade_size_s:-fade_size_s]
        self.assertAlmostEqualVector(m, essentia.array([-20.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-20.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -20., 0.1)

        # M, S, I = -26 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', '1kHz_sine_-26LUFS-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        m = m[fade_size_m:-fade_size_m]
        s = s[fade_size_s:-fade_size_s]
        self.assertAlmostEqualVector(m, essentia.array([-26.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-26.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -26., 0.1)

        # M, S, I = -40 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', '1kHz_sine_-40LUFS-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        m = m[fade_size_m:-fade_size_m]
        s = s[fade_size_s:-fade_size_s]
        self.assertAlmostEqualVector(m, essentia.array([-40.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-40.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -40., 0.1)

        # M, S, I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-1-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        m = m[fade_size_m:-fade_size_m]
        s = s[fade_size_s:-fade_size_s]
        self.assertAlmostEqualVector(m, essentia.array([-23.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-23.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -23., 0.1)

        # M, S, I = -33 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-2-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        m, s, i, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        m = m[fade_size_m:-fade_size_m]
        s = s[fade_size_s:-fade_size_s]
        self.assertAlmostEqualVector(m, essentia.array([-33.] * len(m)), 0.1)
        self.assertAlmostEqualVector(s, essentia.array([-33.] * len(s)), 0.1)
        self.assertAlmostEqual(i, -33., 0.1)

        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-3-16bit-v02.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(i, -23., 0.1)

        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-4-16bit-v02.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(i, -23., 0.1)


        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-5-16bit-v02.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(i, -23., 0.1)

        # I = -23 +- 0.1 LUFS
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3341-7_seq-3342-5-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(i, -23., 0.1)

        # Test audio files for dynamic range are provided in EBU Tech Doc 3342
        # https://tech.ebu.ch/docs/tech/tech3342.pdf

        # LRA = 10 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-1-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, i, _ = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(r, 10., 1.)

        # LRA = 5 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-2-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(r, 5., 1.)

        # LRA = 20 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-3-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(r, 20., 1.)

        # LRA = 15 +- 1 LU
        filename = join(testdata.audio_dir, 'generated', 'ebur128', 'seq-3342-4-16bit.flac')
        audio, samplerate, _, _, _, _ = AudioLoader(filename=filename)()
        _, _, _, r = LoudnessEBUR128(sampleRate=samplerate,
                                     hopSize=0.1,
                                     startAtZero=True)(audio)
        self.assertAlmostEqual(r, 15., 1.)

    def testEmpty(self):
        # empty (0,2) array
        audio = essentia.array([[1., 1.]])[:-1]
        self.assertComputeFails(LoudnessEBUR128(), audio)

    def testSilence(self):
        audio = essentia.array([[0, 0]] * 44100)
        m, s, i, r = LoudnessEBUR128()(audio)

        # Momentary and short-term loudness can have values below absolute threshold of -70. LUFS
        for x in m:
            self.assert_(x <= -70.)
        for x in s:
            self.assert_(x <= -70.)
        self.assertEqual(i, -70.)
        self.assertEqual(r, 0.)

suite = allTests(TestLoudnessEBUR128)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
