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

    # ------------------------------------------------------------------------------------
    # startTime / endTime -- issue #771
    #
    # The reference for all of these is the decode-and-discard path: decode the whole file
    # and cut the slice out afterwards. That is exactly what a caller had to do before, so
    # agreeing with it IS the definition of a correct seek. Slices are cut with Trimmer's
    # seconds -> samples rule, which is the rule the loader uses as well.

    def slice(self, audio, sampleRate, startTime, endTime):
        return numpy.array(audio)[int(startTime*sampleRate):int(endTime*sampleRate)]

    def seeked(self, filename, startTime, endTime):
        from essentia.standard import AudioLoader as stdAudioLoader
        audio, _, _, _, _, _ = stdAudioLoader(filename=filename,
                                              startTime=startTime,
                                              endTime=endTime)()
        return numpy.array(audio)

    def sliceIsExact(self, filename, spans):
        from essentia.standard import AudioLoader as stdAudioLoader
        filename = join(testdata.audio_dir, filename)
        whole, sampleRate, _, _, _, _ = stdAudioLoader(filename=filename)()
        for startTime, endTime in spans:
            found = self.seeked(filename, startTime, endTime)
            expected = self.slice(whole, sampleRate, startTime, endTime)
            self.assertEqual(len(found), len(expected))
            self.assert_(numpy.array_equal(found, expected),
                         'seeking to %s s did not reproduce the decode-and-discard reference'
                         % startTime)

    def testSeekPcm(self):
        spans = [(0., 2.), (1., 3.), (12.345, 14.345), (30., 45.)]
        self.sliceIsExact(join('recorded', 'musicbox.wav'), spans)

    def testSeekFlac(self):
        self.sliceIsExact(join('recorded', 'dubstep.flac'), [(0., 2.), (1.5, 3.5), (4., 6.)])

    def testSeekMp3(self):
        self.sliceIsExact(join('recorded', 'techno_loop.mp3'),
                          [(0., 2.), (1.5, 3.5), (10., 12.), (25., 30.)])

    def testSeekOgg(self):
        self.sliceIsExact(join('recorded', 'dubstep.ogg'), [(0., 2.), (1.5, 3.5), (4., 6.)])

    def testSeekOggOpus(self):
        # Regression test for upstream issue 1531 (MTG/essentia): AudioLoader must set
        # AVCodecContext::pkt_timebase before avcodec_open2(). libavcodec expresses the
        # frame->pts adjustment it makes after discarding a codec's initial padding in
        # pkt_timebase; left at its 0/1 default the adjustment is silently skipped, and on
        # Ogg/Opus every decoded frame's pts is short by exactly the 312-sample pre-skip
        # Opus mandates at 48 kHz. Nothing consumes frame->pts except the post-seek anchor,
        # so the symptom is a slice shifted by exactly +312 samples whenever startTime > 0
        # on an Opus file -- which is what this test pins down.
        #
        # The test-audio collection has no Opus file, so the fixture is generated here with
        # the ffmpeg CLI: a phase-continuous 997 Hz sine at Opus's native 48 kHz, on which
        # any misalignment produces an unambiguous, large amplitude error. Skipped cleanly
        # when no ffmpeg CLI or no Opus encoder is available.
        import os
        import shutil
        import subprocess
        import tempfile

        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            self.skipTest('no ffmpeg CLI available to generate an Ogg/Opus fixture')

        tmpdir = tempfile.mkdtemp(suffix='_essentia_opus')
        try:
            filename = os.path.join(tmpdir, 'sine48k.opus')
            source = ['-f', 'lavfi',
                      '-i', 'sine=frequency=997:sample_rate=48000:duration=10']
            encoders = [['-c:a', 'libopus', '-b:a', '128k'],
                        ['-c:a', 'opus', '-strict', 'experimental', '-b:a', '128k']]
            for encoder in encoders:
                result = subprocess.run([ffmpeg, '-v', 'error', '-y'] + source +
                                        encoder + [filename],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    break
            else:
                self.skipTest('ffmpeg CLI has no usable Opus encoder')

            from essentia.standard import AudioLoader as stdAudioLoader
            whole, sampleRate, _, _, _, _ = stdAudioLoader(filename=filename)()
            self.assertEqual(sampleRate, 48000)

            for startTime, endTime in [(5., 7.), (1.5, 3.5)]:
                found = self.seeked(filename, startTime, endTime)
                expected = self.slice(whole, sampleRate, startTime, endTime)
                self.assertEqual(len(found), len(expected))
                # Not array_equal: the Opus decoder's post-seek convergence leaves a 1-ULP
                # (~1.5e-8) residual over the first ~60 ms of a slice, which is the codec's
                # doing, not the loader's. 2e-7 absorbs that and nothing else: a slice
                # shifted by the 312-sample pre-skip differs from the reference by O(1)
                # amplitude on this sine, seven orders of magnitude above the bound.
                maxdiff = numpy.max(numpy.abs(found.astype(numpy.float64) -
                                              expected.astype(numpy.float64)))
                self.assert_(maxdiff <= 2e-7,
                             'seeking to %s s in an Ogg/Opus file did not reproduce the '
                             'decode-and-discard reference (max amplitude error %g, '
                             'misaligned by %s samples); is AVCodecContext::pkt_timebase '
                             'set? (upstream issue 1531)'
                             % (startTime, maxdiff, self.measuredLag(found, expected)))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def measuredLag(self, found, expected, maxLag=1000):
        # Diagnostic only (used in failure messages): the lag k minimizing the mean absolute
        # difference between found[k:...] and expected, scanned over +/- maxLag samples.
        # A pkt_timebase regression shows up here as exactly the Opus pre-skip, +312.
        a = numpy.array(found)[:, 0].astype(numpy.float64)
        b = numpy.array(expected)[:, 0].astype(numpy.float64)
        n = min(len(a), len(b)) - maxLag
        if n <= maxLag: return 'unknown'
        best, bestErr = 0, None
        for k in range(-maxLag, maxLag + 1):
            if k >= 0: err = numpy.abs(a[k:k+n-maxLag] - b[:n-maxLag]).mean()
            else:      err = numpy.abs(a[:n-maxLag] - b[-k:-k+n-maxLag]).mean()
            if bestErr is None or err < bestErr: best, bestErr = k, err
        return best

    def testSeekAac(self):
        # AAC is the one format a seek cannot reproduce exactly, and it is the codec's doing,
        # not the loader's: Perceptual Noise Substitution synthesises noise bands from a PRNG
        # whose state depends on the whole decode history, so a decoder that starts anywhere
        # but at sample 0 fills those bands with different noise. No amount of preroll fixes
        # it (the residual is flat from 0.05 s to 10 s of preroll) and it disappears entirely
        # when the same audio is encoded with -aac_pns 0. The affected bands are synthesised
        # noise by construction, so this is a documented expectation, not a defect.
        from essentia.standard import AudioLoader as stdAudioLoader
        filename = join(testdata.audio_dir, 'recorded', 'dubstep.aac')
        whole, sampleRate, _, _, _, _ = stdAudioLoader(filename=filename)()
        for startTime, endTime in [(1.5, 3.5), (4., 6.)]:
            found = self.seeked(filename, startTime, endTime)
            expected = self.slice(whole, sampleRate, startTime, endTime)
            self.assertEqual(len(found), len(expected))
            self.assertAlmostEqualVectorAbs(found.flatten(), expected.flatten(), 5e-2)

    def testSeekBoundaries(self):
        from essentia.standard import AudioLoader as stdAudioLoader
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole, sampleRate, _, _, _, _ = stdAudioLoader(filename=filename)()
        duration = len(whole) / float(sampleRate)

        # startTime 0 with no endTime is the default, and must still read the whole file
        self.assert_(numpy.array_equal(self.seeked(filename, 0., 1e6), numpy.array(whole)))

        # an endTime past the end of the file simply stops at the end of the file
        found = self.seeked(filename, duration - 1., duration + 100.)
        self.assert_(numpy.array_equal(found, self.slice(whole, sampleRate, duration - 1., 1e6)))

        # so does an endTime landing exactly on it
        self.assert_(numpy.array_equal(self.seeked(filename, 0., duration), numpy.array(whole)))

        # a startTime past the end of the file yields nothing at all, it is not an error
        self.assertEqual(len(self.seeked(filename, duration + 10., duration + 20.)), 0)

        # neither is an empty slice: startTime == endTime yields no samples
        self.assertEqual(len(self.seeked(filename, 0., 0.)), 0)
        self.assertEqual(len(self.seeked(filename, 1., 1.)), 0)

        # an INVERTED slice is an error, as it has always been for the algorithms that
        # expose these parameters
        self.assertConfigureFails(sAudioLoader(), {'filename': filename,
                                                   'startTime': 10., 'endTime': 1.})
        self.assertConfigureFails(sAudioLoader(), {'filename': filename, 'startTime': -1.})
        self.assertConfigureFails(sAudioLoader(), {'filename': filename, 'endTime': -1.})

    def testSeekShortFile(self):
        # a file shorter than the seek preroll must still come back with the right audio
        from essentia.standard import AudioLoader as stdAudioLoader
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        whole, sampleRate, _, _, _, _ = stdAudioLoader(filename=filename)()
        for startTime, endTime in [(0.1, 0.2), (2., 3.), (0.25, 1e6)]:
            found = self.seeked(filename, startTime, endTime)
            expected = self.slice(whole, sampleRate, startTime, endTime)
            self.assertEqual(len(found), len(expected))
            self.assert_(numpy.array_equal(found, expected))

    def testSeekedMD5(self):
        # a slice does not read the whole payload, so its md5 cannot be the file's md5;
        # report nothing rather than a checksum that means nothing.
        from essentia.standard import AudioLoader as stdAudioLoader
        filename = join(testdata.audio_dir, 'recorded', 'dubstep.wav')
        _, _, _, md5, _, _ = stdAudioLoader(filename=filename, computeMD5=True,
                                            startTime=1., endTime=2.)()
        self.assertEqual(md5, '')

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
        self.assertEqual(len(audio1), 441000)
        self.assertEqual(len(audio2), 441000)
        self.assertEqual(len(audio3), 441000)
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

        directory = join(testdata.audio_dir,'recorded')
        _, _, _, md5_wav, _, _ = AudioLoader(filename=join(directory,"dubstep.wav"), computeMD5=True)()
        _, _, _, md5_flac, _, _ = AudioLoader(filename=join(directory,"dubstep.flac"), computeMD5=True)()
        _, _, _, md5_mp3, _, _ = AudioLoader(filename=join(directory,"dubstep.mp3"), computeMD5=True)()
        _, _, _, md5_ogg, _, _ = AudioLoader(filename=join(directory,"dubstep.ogg"), computeMD5=True)()
        _, _, _, md5_aac, _, _ = AudioLoader(filename=join(directory,"dubstep.aac"), computeMD5=True)()

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
