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
            print 'WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...'
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
        audiofile = join(testdata.audio_dir,'recorded','musicbox.wav')
        loader = stdAudioLoader(filename=audiofile)
        audio1, sr1, nChannels1 = loader();
        audio2, sr2, nchannels2 = loader();
        loader.reset();
        audio3, sr3, nChannels3 = loader();
        self.assertAlmostEqualMatrix(audio3, audio1)
        self.assertEqual(sr3, sr1)
        self.assertEqual(nChannels3, nChannels1)
        self.assertEqualMatrix(audio2, audio1)

    def testLoadMultiple(self):
        from essentia.standard import AudioLoader as stdAudioLoader
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = stdAudioLoader(filename=filename)
        audio1, _, _ = algo()
        audio2, _, _ = algo()
        audio3, _, _ = algo()
        self.assertEquals(len(audio1), 441000);
        self.assertEquals(len(audio2), 441000);
        self.assertEquals(len(audio3), 441000);
        self.assertEqualMatrix(audio2, audio1)
        self.assertEqualMatrix(audio2, audio3)

    def testBitrate(self):
        from math import fabs
        dir = join(testdata.audio_dir,'recorded')
        audio16, sr16, ch16 = AudioLoader(filename=join(dir,"cat_purrrr.wav"))()
        audio24, sr24, ch24 = AudioLoader(filename=join(dir,"cat_purrrr24bit.wav"))()
        audio32, sr32, ch32 = AudioLoader(filename=join(dir,"cat_purrrr32bit.wav"))()
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



suite = allTests(TestAudioLoader_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
