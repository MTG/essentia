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
import essentia
from essentia.streaming import AudioLoader as sAudioLoader
import sys


class TestAudioLoader_Streaming(TestCase):

    def pcm(self, sampleRate, filename, stereo=False):
        # as comparing sample by sample will probably fail the results, in this test
        # a filename that lasts 10s with impulses of 1 every second is loaded.
        # the sum of the impulses computed and compared with the expectedSum value.
        loader = sAudioLoader(filename=join(testdata.audio_dir, filename))
        p = Pool()

        loader.audio >> (p, 'audio')
        loader.numberChannels >> (p, 'nChannels')
        loader.sampleRate >> (p, 'sampleRate')
        loader.md5 >> (p, 'md5')
        loader.bit_rate >> (p, 'bit_rate')
        loader.codec >> (p, 'codec')

        run(loader)

        self.assertEqual(p['sampleRate'], sampleRate)
        if stereo: self.assertEqual(p['nChannels'], 2)
        else:      self.assertEqual(p['nChannels'], 1)

        audio = p['audio']

        self.assertEqual(len(audio), 10*sampleRate)

        sum = 0

        # compute sum
        for stereoSample in audio:
            sum += stereoSample[0] + stereoSample[1]

        if stereo: self.assertAlmostEqual(sum, 18, 1e-3)
        else:      self.assertAlmostEqual(sum, 9, 1e-3)


    def compressed(self, sampleRate, filename, stereo=False):
        # for compressed files we will compare only those values above certain threshold.
        noisefloor = 10.0/32767.0

        loader = sAudioLoader(filename=join(testdata.audio_dir, filename))
        p = Pool()
        loader.audio >> (p, 'audio')
        loader.numberChannels >> (p, 'nChannels')
        loader.sampleRate >> (p, 'sampleRate')
        loader.md5 >> (p, 'md5')
        loader.bit_rate >> (p, 'bit_rate')
        loader.codec >> (p, 'codec')

        run(loader)

        if stereo: self.assertEqual(p['nChannels'], 2)
        else:      self.assertEqual(p['nChannels'], 1)
        self.assertEqual(p['sampleRate'], sampleRate)

        sum = 0

        for stereoSample in p['audio']:
            left = abs(stereoSample[0])
            right = abs(stereoSample[1])

            # don't add absolute values!
            if left > noisefloor: sum += round(stereoSample[0])
            if right > noisefloor: sum += round(stereoSample[1])

        # TODO: ffmpeg seems to decode ogg files in opposite phase, thus:
        if sum < 0:
            print('WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...')
            sum = abs(sum)

        if stereo: self.assertEqual(sum, 18)
        else:      self.assertEqual(sum, 9)

    def testPcm(self):
        # .wav
        wavpath = join('generated','synthesised','impulse','wav')
        self.pcm(44100, join(wavpath,'impulses_1second_44100.wav'))
        self.pcm(44100, join(wavpath,'impulses_1second_44100_st.wav'), True)
        self.pcm(48000, join(wavpath,'impulses_1second_48000.wav'))
        self.pcm(48000, join(wavpath,'impulses_1second_48000_st.wav'), True)
        self.pcm(22050, join(wavpath,'impulses_1second_22050.wav'))
        self.pcm(22050, join(wavpath,'impulses_1second_22050_st.wav'), True)

        # .aiff
        aiffpath = join('generated','synthesised','impulse','aiff')
        self.pcm(44100, join(aiffpath,'impulses_1second_44100.aiff'))
        self.pcm(44100, join(aiffpath,'impulses_1second_44100_st.aiff'), True)
        self.pcm(48000, join(aiffpath,'impulses_1second_48000.aiff'))
        self.pcm(48000, join(aiffpath,'impulses_1second_48000_st.aiff'), True)
        self.pcm(22050, join(aiffpath,'impulses_1second_22050.aiff'))
        self.pcm(22050, join(aiffpath,'impulses_1second_22050_st.aiff'), True)

    def testOgg(self):
        oggpath = join('generated','synthesised','impulse','ogg')
        self.compressed(44100, join(oggpath, 'impulses_1second_44100.ogg'))
        self.compressed(44100, join(oggpath, 'impulses_1second_44100_st.ogg'), True)
        self.compressed(48000, join(oggpath, 'impulses_1second_48000.ogg'))
        self.compressed(48000, join(oggpath, 'impulses_1second_48000_st.ogg'), True)
        self.compressed(22050, join(oggpath, 'impulses_1second_22050.ogg'))
        self.compressed(22050, join(oggpath, 'impulses_1second_22050_st.ogg'), True)

    def testFlac(self):
        flacpath = join('generated','synthesised','impulse','flac')
        self.pcm(44100, join(flacpath, 'impulses_1second_44100.flac'))
        self.pcm(44100, join(flacpath, 'impulses_1second_44100_st.flac'), True)
        self.pcm(48000, join(flacpath, 'impulses_1second_48000.flac'))
        self.pcm(48000, join(flacpath, 'impulses_1second_48000_st.flac'), True)
        self.pcm(22050, join(flacpath, 'impulses_1second_22050.flac'))
        self.pcm(22050, join(flacpath, 'impulses_1second_22050_st.flac'), True)

    def testMp3(self):
        mp3path = join('generated','synthesised','impulse','mp3')
        self.compressed(44100, join(mp3path, 'impulses_1second_44100.mp3'))
        self.compressed(44100, join(mp3path, 'impulses_1second_44100_st.mp3'), True)
        self.compressed(48000, join(mp3path, 'impulses_1second_48000.mp3'))
        self.compressed(48000, join(mp3path, 'impulses_1second_48000_st.mp3'), True)
        self.compressed(22050, join(mp3path, 'impulses_1second_22050.mp3'))
        self.compressed(22050, join(mp3path, 'impulses_1second_22050_st.mp3'), True)

    def testInvalidFile(self):
        for ext in ['wav', 'aiff', 'flac', 'mp3', 'ogg']:
            self.assertRaises(RuntimeError, lambda: sAudioLoader(filename='unknown.'+ext))

    def testMultiChannel(self):
        for ext in ['wav', 'aiff', 'flac']:
            filename = join(testdata.audio_dir, 'generated', 'multichannel', '4channels.'+ext)
            self.assertRaises(RuntimeError, lambda: sAudioLoader(filename=filename))

    def testResetStandard(self):
        from essentia.standard import AudioLoader as stdAudioLoader
        audiofile = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        loader = stdAudioLoader(filename=audiofile, computeMD5=True)
        audio1, sr1, nChannels1, md51, bitrate1, codec1 = loader()
        audio2, sr2, nchannels2, md52, bitrate2, codec2 = loader()
        loader.reset()
        audio3, sr3, nChannels3, md53, bitrate3, codec3 = loader()
        self.assertAlmostEqualMatrix(audio3, audio1)
        self.assertEqual(sr3, sr1)
        self.assertEqual(nChannels3, nChannels1)
        self.assertEqual(md53, md51)
        self.assertEqualMatrix(audio2, audio1)
        self.assertEqual(bitrate3, bitrate1)
        self.assertEqual(codec3, codec1)

    def testLoadMultiple(self):
        from essentia.standard import AudioLoader as stdAudioLoader
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = stdAudioLoader(filename=filename)
        audio1, _, _, _, _, _ = algo()
        audio2, _, _, _, _, _ = algo()
        audio3, _, _, _, _, _ = algo()
        self.assertEquals(len(audio1), 441000);
        self.assertEquals(len(audio2), 441000);
        self.assertEquals(len(audio3), 441000);
        self.assertEqualMatrix(audio2, audio1)
        self.assertEqualMatrix(audio2, audio3)

    def testBitrate(self):
        from math import fabs
        dir = join(testdata.audio_dir,'recorded')
        audio16, sr16, ch16, md516, _, _ = AudioLoader(filename=join(dir,"cat_purrrr.wav"))()
        audio24, sr24, ch24, md524, _, _ = AudioLoader(filename=join(dir,"cat_purrrr24bit.wav"))()
        audio32, sr32, ch32, md532, _, _ = AudioLoader(filename=join(dir,"cat_purrrr32bit.wav"))()
        audio16L, audio16R = essentia.standard.StereoDemuxer()(audio16)
        audio24L, audio24R = essentia.standard.StereoDemuxer()(audio24)
        audio32L, audio32R = essentia.standard.StereoDemuxer()(audio32)

        error24 = 0
        for i, j in zip(audio16L, audio24L): error24 += fabs(fabs(i) - fabs(j))
        for i, j in zip(audio16R, audio24R): error24 += fabs(fabs(i) - fabs(j))

        error32 = 0
        for i, j in zip(audio16L, audio32L): error32 += fabs(fabs(i) - fabs(j))
        for i, j in zip(audio16R, audio32R): error32 += fabs(fabs(i) - fabs(j))

        sum16 = sum(audio16L) + sum(audio16R)
        sum24 = sum(audio24L) + sum(audio24R)
        sum32 = sum(audio32L) + sum(audio32R)

        centroid = essentia.standard.Centroid()
        centroid16 = centroid(audio16L)
        centroid24 = centroid(audio24L)
        centroid32 = centroid(audio32L)

        self.assertEqual(len(audio16), len(audio24))
        self.assertEqual(len(audio16), len(audio32))
        self.assertAlmostEqual(error24, 0)
        self.assertAlmostEqual(error32, 0)
        self.assertAlmostEqual(sum16-sum24, 0)
        self.assertAlmostEqual(sum16-sum32, 0)
        self.assertAlmostEqual(centroid16-centroid24, 0)
        self.assertAlmostEqual(centroid16-centroid32, 0)

    def testMD5(self):

        dir = join(testdata.audio_dir,'recorded')
        _, _, _, md5_wav, _, _ = AudioLoader(filename=join(dir,"dubstep.wav"), computeMD5=True)()
        _, _, _, md5_flac, _, _ = AudioLoader(filename=join(dir,"dubstep.flac"), computeMD5=True)()
        _, _, _, md5_mp3, _, _ = AudioLoader(filename=join(dir,"dubstep.mp3"), computeMD5=True)()
        _, _, _, md5_ogg, _, _ = AudioLoader(filename=join(dir,"dubstep.ogg"), computeMD5=True)()
        _, _, _, md5_aac, _, _ = AudioLoader(filename=join(dir,"dubstep.aac"), computeMD5=True)()

        # results should correspond to ffmpeg output (computed on debian wheezy)
        #   ffmpeg -i dubstep.wav -acodec copy -f md5 -
        self.assertEqual(md5_wav, "bf0f4d0613fab0fa5268ece9b043c441")
        self.assertEqual(md5_flac, "93ee45bc8776eed656a554b32d0d9616")
        self.assertEqual(md5_mp3, "1e5a598218e9b19cfe04d6c2f61f84a6")
        self.assertEqual(md5_ogg, "a87dad40fea0966cc5b967d5412e8868")
        self.assertEqual(md5_aac, "9a4c7f0da68d4b58767f219c48014f9c")

    def testMultiStream(self):

        #  stream 0 of multistream1.mka is the same as stream 1 of multistream2.mka

        p = Pool()

        stream0 = sAudioLoader(filename=join(testdata.audio_dir, 'generated', 'multistream', 'multistream1.mka'), audioStream=0)
        stream1 = sAudioLoader(filename=join(testdata.audio_dir, 'generated', 'multistream', 'multistream2.mka'), audioStream=1)

        stream0.audio >> (p, 'stream0')
        stream0.numberChannels >> (p, 'nChannels0')
        stream0.sampleRate >> (p, 'sampleRate0')
        stream0.md5 >> (p, 'md50')
        stream0.bit_rate >> (p, 'bit_rate0')
        stream0.codec >> (p, 'codec0')

        stream1.audio >> (p, 'stream1')
        stream1.numberChannels >> (p, 'nChannels1')
        stream1.sampleRate >> (p, 'sampleRate1')
        stream1.md5 >> (p, 'md51')
        stream1.bit_rate >> (p, 'bit_rate1')
        stream1.codec >> (p, 'codec1')

        run(stream0)
        run(stream1)

        self.assertEqualVector(p['stream0'][0], p['stream1'][0])

        # An exception should be thrown if the required audioStream is out of bounds
        self.assertConfigureFails(sAudioLoader(), {'filename': join(testdata.audio_dir, 'generated', 'multistream', 'multistream1.mka'), 'audioStream': 2})


suite = allTests(TestAudioLoader_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
