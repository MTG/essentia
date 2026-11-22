#!/usr/bin/env python


#
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
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/


from essentia_test import *
from numpy import fabs
audio_dir = join(testdata.audio_dir, 'generated', 'synthesised', 'impulse')
wav_dir = join(audio_dir, 'wav')
ogg_dir = join(audio_dir, 'ogg')
mp3_dir = join(audio_dir, 'mp3')
resamp_dir = join(audio_dir, 'resample')

class TestMonoLoader(TestCase):

    def round(self, val):
        if val >= 0 : return int(val+0.5)
        return int(val-0.5)

    def load(self, filename, downmix, sampleRate):
        return MonoLoader(filename=filename, downmix=downmix, sampleRate=sampleRate)()

    def testInvalidParam(self):
        filename = join(wav_dir, 'impulses_1second_44100_st.wav')
        cfg = {'filename': filename, 'downmix': 'stereo', 'sampleRate': 44100}
        self.assertConfigureFails(MonoLoader(sampleRate=44100), cfg)

        cfg = {'filename': filename, 'downmix': 'left', 'sampleRate': 0}
        self.assertConfigureFails(MonoLoader(sampleRate=44100), cfg)

        filename = 'unknown.wav'
        cfg = {'filename': filename, 'downmix': 'left', 'sampleRate': 44100}
        self.assertConfigureFails(MonoLoader(), cfg)


    def testWav44100(self):
        # files with 9 impulses in each channel
        filename = join(wav_dir, 'impulses_1second_44100_st.wav')
        left = self.load(filename, 'left', 44100)
        right = self.load(filename, 'right', 44100)
        mix = self.load(filename, 'mix', 44100)
        self.assertEqual(self.round(sum(left)), 9)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(self.round(sum(mix)), 9)

    def testWav22050(self):
        # files with 9 impulses in each channel
        filename = join(wav_dir, 'impulses_1second_22050_st.wav')
        left = self.load(filename, 'left', 22050)
        right = self.load(filename, 'right', 22050)
        mix = self.load(filename, 'mix', 22050)
        self.assertEqual(self.round(sum(left)), 9)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(self.round(sum(mix)), 9)

    def testWav48000(self):
        # files with 9 impulses in each channel
        filename = join(wav_dir, 'impulses_1second_48000_st.wav')
        left = self.load(filename, 'left', 48000)
        right = self.load(filename, 'right', 48000)
        mix = self.load(filename, 'mix', 48000)
        self.assertEqual(self.round(sum(left)), 9)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(self.round(sum(mix)), 9)

    def testEmptyWav(self):
        filename = join(testdata.audio_dir, 'generated', 'empty', 'empty.aiff')
        self.assertEqualVector(MonoLoader(filename=filename, downmix='left', sampleRate=44100)(), [])

    def testWavLeftRightOffset(self):
        # file with 9 impulses in right channel and 10 in left channel
        dir = join(testdata.audio_dir, 'generated', 'synthesised', 'impulse', 'left_right_offset')
        filename = join(dir, 'impulses_1second_44100.wav')
        left = self.load(filename, 'left', 44100)
        right = self.load(filename, 'right', 44100)
        mix = self.load(filename, 'mix', 44100)
        self.assertEqual(self.round(sum(left)), 10)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertAlmostEqualFixedPrecision(sum(mix), 9.5, 3) # 0.5*left + 0.5*right

###############
# #mp3
###############

    def sum(self, l):
        result = 0.0
        noisefloor = 0.003
        for i in range(len(l)):
            if fabs(l[i]) > noisefloor:
                result+= l[i]
        return self.round(result)

    def testMp344100(self):
        # files with 9 impulses in each channel
        filename = join(mp3_dir, 'impulses_1second_44100_st.mp3')
        left = self.load(filename, 'left', 44100)
        right = self.load(filename, 'right', 44100)
        mix = self.load(filename, 'mix', 44100)

        self.assertEqual(self.sum(left), 9)
        self.assertEqual(self.sum(right), 9)
        self.assertEqual(self.sum(mix), 9)

    def testMp322050(self):
        # files with 9 impulses in each channel
        filename = join(mp3_dir, 'impulses_1second_22050_st.mp3')
        left = self.load(filename, 'left', 22050)
        right = self.load(filename, 'right', 22050)
        mix = self.load(filename, 'mix', 22050)

        self.assertEqual(self.sum(left), 9)
        self.assertEqual(self.sum(right), 9)
        self.assertEqual(self.sum(mix), 9)

    def testMp348000(self):
        # files with 9 impulses in each channel
        filename = join(mp3_dir, 'impulses_1second_48000_st.mp3')
        left = self.load(filename, 'left', 48000)
        right = self.load(filename, 'right', 48000)
        mix = self.load(filename, 'mix', 48000)
        self.assertEqual(self.sum(left), 9)
        self.assertEqual(self.sum(right), 9)
        self.assertEqual(self.sum(mix), 9)

    def testMp3TimeShift(self):
        # test mp3s are loaded with no time shift (lost frames)
        filename_mp3 = join(mp3_dir, 'impulses_1second_44100.mp3')
        filename_wav = join(wav_dir, 'impulses_1second_44100.wav')
        mp3 = self.load(filename_mp3, 'mix', 44100)
        wav = self.load(filename_wav, 'mix', 44100)

        # find time shift between impulse positions
        impulses_mp3 = [x for x in range(len(mp3)) if mp3[x]>0.9]
        impulses_wav = [x for x in range(len(wav)) if wav[x]>0.9]

        shift = impulses_mp3[0] - impulses_wav[0]
        # FIXME:
        # For this particular audio files in essentia 2.1_beta2 with an older libav version
        # the expected shift was 1105 samples, however now there is no shift
        # Nevertheless time shift can be observed on other examples but we still do not have such tests 

        #self.assertEqual(abs(shift), 1105)
        self.assertEqual(abs(shift), 0)


###############
# #OGG
###############

    def testOgg44100(self):
        filename = join(ogg_dir, 'impulses_1second_44100_st.ogg')
        left = self.load(filename, 'left', 44100)
        right = self.load(filename, 'right', 44100)
        mix = self.load(filename, 'mix', 44100)
        self.assertEqual(abs(self.sum(left)),  9)
        self.assertEqual(abs(self.sum(right)), 9)
        self.assertEqual(abs(self.sum(mix)),   9)

        if self.sum(left) < 0:
            print('WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...')

    def testOgg22050(self):
        # files with 9 impulses in each channel
        filename = join(ogg_dir, 'impulses_1second_22050_st.ogg')
        left = self.load(filename, 'left', 22050)
        right = self.load(filename, 'right', 22050)
        mix = self.load(filename, 'mix', 22050)
        self.assertEqual(abs(self.sum(left)),  9)
        self.assertEqual(abs(self.sum(right)), 9)
        self.assertEqual(abs(self.sum(mix)),   9)

        if self.sum(left) < 0:
            print('WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...')

    def testOgg48000(self):
        # files with 9 impulses in each channel
        filename = join(ogg_dir, 'impulses_1second_48000_st.ogg')
        left = self.load(filename, 'left', 48000)
        right = self.load(filename, 'right', 48000)
        mix = self.load(filename, 'mix', 48000)
        self.assertEqual(abs(self.sum(left)),  9)
        self.assertEqual(abs(self.sum(right)), 9)
        self.assertEqual(abs(self.sum(mix)),   9)

        if self.sum(left) < 0:
            print('WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...')

    def testDownSampling(self):
        # files of 30s with impulses at every sample
        # from 44100 to 22050
        filename = join(resamp_dir, 'impulses_1samp_44100.wav')
        left = self.load(filename, 'left', 22050)
        self.assertAlmostEqual(sum(left), 30.*22050, 1e-4)
        # from 48000 to 44100
        filename = join(resamp_dir, 'impulses_1samp_48000.wav')
        left = self.load(filename, 'left', 44100)
        self.assertAlmostEqual(sum(left), 30.*44100, 1e-4)
        # from 48000 to 22050
        left = self.load(filename, 'left', 22050)
        self.assertAlmostEqual(sum(left), 30.*22050, 1e-4)

    def testUpSampling(self):
        # from 44100 to 48000
        filename = join(resamp_dir, 'impulses_1samp_44100.wav')
        left = self.load(filename, 'right', 48000)
        self.assertAlmostEqual(sum(left), 30.*48000, 1e-4)
        # from 22050 to 44100
        filename = join(resamp_dir, 'impulses_1samp_22050.wav')
        left = self.load(filename, 'right', 44100)
        self.assertAlmostEqual(sum(left), 30.*44100, 1e-4)
        # from 22050 to 48000
        left = self.load(filename, 'right', 48000)
        self.assertAlmostEqual(sum(left), 30.*48000, 1e-4)

    def testInvalidFilename(self):
        self.assertConfigureFails(MonoLoader(),{'filename':'unknown.wav'})

    # ------------------------------------------------------------------------------------
    # startTime / endTime -- issue #771. MonoLoader forwards them to AudioLoader, which
    # seeks. The reference is the decode-and-discard path: load the whole file and cut the
    # slice out afterwards, using Trimmer's seconds -> samples rule.

    def slice(self, audio, sampleRate, startTime, endTime):
        return numpy.array(audio)[int(startTime*sampleRate):int(endTime*sampleRate)]

    def testSeek(self):
        # at the file's own rate Resample is a fastcopy, so the seeked read is EXACT
        for name, sampleRate in [(join('recorded', 'musicbox.wav'), 44100),
                                 (join('recorded', 'techno_loop.mp3'), 44100),
                                 (join('recorded', 'dubstep.flac'), 44100),
                                 (join('recorded', 'guitar_triads.flac'), 48000)]:
            filename = join(testdata.audio_dir, name)
            whole = MonoLoader(filename=filename, sampleRate=sampleRate)()
            for startTime, endTime in [(0., 2.), (1.5, 3.5), (5., 6.)]:
                found = numpy.array(MonoLoader(filename=filename, sampleRate=sampleRate,
                                               startTime=startTime, endTime=endTime)())
                expected = self.slice(whole, sampleRate, startTime, endTime)
                self.assertEqual(len(found), len(expected))
                self.assert_(numpy.array_equal(found, expected),
                             '%s: seeked read differs from the reference at %s s'
                             % (name, startTime))

    def testSeekResampled(self):
        # When the loader also RESAMPLES, the seeked read is close but not identical, and
        # the reason is libsamplerate rather than the seek: its output depends on how much
        # input it has already consumed (filter history plus a phase accumulator), so a
        # converter started at startTime is not in the same state as one started at 0. The
        # difference is uniform across the slice rather than a startup transient, and no
        # bounded amount of preroll removes it. This test pins the size of that residual so
        # a regression in the seek itself -- which would be far larger -- is still caught.
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole = MonoLoader(filename=filename, sampleRate=22050)()
        for startTime, endTime in [(1.5, 3.5), (5., 6.)]:
            found = numpy.array(MonoLoader(filename=filename, sampleRate=22050,
                                           startTime=startTime, endTime=endTime)())
            expected = self.slice(whole, 22050, startTime, endTime)
            # the length may differ by a sample, as it does between architectures
            self.assert_(abs(len(found) - len(expected)) <= 1)
            n = min(len(found), len(expected))
            residual = numpy.sqrt(numpy.mean((found[:n] - expected[:n])**2.))
            reference = numpy.sqrt(numpy.mean(expected[:n]**2.))
            self.assert_(residual < reference/100.,
                         'resampled seek residual %e is too large against %e'
                         % (residual, reference))

    def testSeekBoundaries(self):
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole = MonoLoader(filename=filename)()
        duration = len(whole)/44100.

        self.assertEqualVector(MonoLoader(filename=filename, startTime=0., endTime=1e6)(), whole)
        self.assertEqual(len(MonoLoader(filename=filename, startTime=duration + 10.,
                                        endTime=duration + 20.)()), 0)
        self.assertEqual(len(MonoLoader(filename=filename, startTime=2., endTime=2.)()), 0)
        self.assertConfigureFails(MonoLoader(), {'filename': filename,
                                                 'startTime': 10., 'endTime': 1.})

    def testResetStandard(self):
        audiofile = join(testdata.audio_dir,'recorded','musicbox.wav')
        loader = MonoLoader(filename=audiofile)
        audio1 = loader()
        audio2 = loader()
        loader.reset()
        audio3 = loader()
        self.assertAlmostEqualVector(audio3, audio1)
        self.assertEqualVector(audio2, audio1)

    def testLoadMultiple(self):
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = MonoLoader(filename=filename)
        audio1 = algo()
        audio2 = algo()
        audio3 = algo()
        self.assertEqual(len(audio1), 441000)
        self.assertEqual(len(audio2), 441000)
        self.assertEqual(len(audio3), 441000)
        self.assertEqualVector(audio2, audio1)
        self.assertEqualVector(audio2, audio3)

suite = allTests(TestMonoLoader)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
