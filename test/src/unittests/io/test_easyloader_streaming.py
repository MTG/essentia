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
from essentia.streaming import EasyLoader, MonoLoader
import sys
import math
class TestEasyLoader_Streaming(TestCase):

    def load(self, inputSampleRate, outputSampleRate,
                   filename, downmix, replayGain, startTime, endTime):
        #for this test we use audio files which have impulses at every sample.
        #files last 30s, longer than 10s, so the resampling is more accurate

        scale = math.pow(10,(replayGain+6.0)/20.0)
        if scale > 1: scale = 1

        loader = EasyLoader(filename=filename,
                            sampleRate = outputSampleRate,
                            downmix = downmix,
                            startTime = startTime,
                            endTime = endTime,
                            replayGain = replayGain)
        pool = Pool()

        loader.audio >> (pool, 'audio')
        run(loader)

        length = int((endTime-startTime)*outputSampleRate)
        # it is kinda weird, some archs (64bit) produce 1 sample more when resampling
        # than what is expected. It is not that much of a problem, though, so we accept
        # it as a correct result
        self.assert_(len(pool['audio']) == length or
                     len(pool['audio']) == length + 1)
        self.assertAlmostEqual(sum(pool['audio']), length*scale, 1e-3)


    def testNoResample(self):
        filename =join(testdata.audio_dir,'generated','synthesised','impulse','resample', 'impulses_1samp_44100.wav')
        self.load(44100, 44100, filename, "left" , 0., 0., 10.)
        self.load(44100, 44100, filename, "right", -15., 3.34, 5.68)
        self.load(44100, 44100, filename, "mix"  , 30., 0.168, 8.32)

    def testResample(self):
        filename = join(testdata.audio_dir, 'generated','synthesised','impulse','resample',
                        'impulses_1samp_44100.wav')
        self.load(44100, 22050, filename, "left" , 0., 0., 10.)
        self.load(44100, 48000, filename, "right", -15., 3.34, 5.68)
        self.load(44100, 11025, filename, "mix"  , 30., 0.168, 8.32)

    # ------------------------------------------------------------------------------------
    # startTime / endTime are no longer applied by a Trimmer after decoding everything: the
    # loader seeks (issue #771). That has to be invisible to callers, so every test below
    # compares against the decode-and-discard path -- read the whole file, cut the slice out
    # afterwards -- which is what this algorithm used to compute.

    def readSlice(self, filename, sampleRate, startTime, endTime, replayGain=-6.):
        loader = EasyLoader(filename=filename, sampleRate=sampleRate, downmix='mix',
                            startTime=startTime, endTime=endTime, replayGain=replayGain)
        pool = Pool()
        loader.audio >> (pool, 'audio')
        run(loader)
        if 'audio' not in pool.descriptorNames():
            return numpy.array([], dtype='float32')
        return numpy.array(pool['audio'])

    def slice(self, audio, sampleRate, startTime, endTime):
        return audio[int(startTime*sampleRate):int(endTime*sampleRate)]

    def sliceIsExact(self, name, sampleRate, spans):
        filename = join(testdata.audio_dir, name)
        whole = self.readSlice(filename, sampleRate, 0., 1e6)
        for startTime, endTime in spans:
            found = self.readSlice(filename, sampleRate, startTime, endTime)
            expected = self.slice(whole, sampleRate, startTime, endTime)
            self.assertEqual(len(found), len(expected))
            self.assert_(numpy.array_equal(found, expected),
                         '%s: the slice [%s, %s) is not the one a full read would give'
                         % (name, startTime, endTime))

    def testSeekPcm(self):
        self.sliceIsExact(join('recorded', 'musicbox.wav'), 44100,
                          [(0., 2.), (1.5, 3.5), (12.345, 14.345), (30., 45.)])

    def testSeekMp3(self):
        self.sliceIsExact(join('recorded', 'techno_loop.mp3'), 44100,
                          [(0., 2.), (1.5, 3.5), (10., 12.), (25., 30.)])

    def testSeekFlac(self):
        self.sliceIsExact(join('recorded', 'dubstep.flac'), 44100,
                          [(0., 2.), (1.5, 3.5), (4., 6.)])

    def testSeekOgg(self):
        self.sliceIsExact(join('recorded', 'dubstep.ogg'), 44100,
                          [(0., 2.), (1.5, 3.5), (4., 6.)])

    def testSeekResampledIsUnchanged(self):
        # A resampled slice is NOT seeked, deliberately: libsamplerate's output depends on
        # how much input it has already consumed, so a converter started at startTime does
        # not reproduce one started at 0 (about -49 dB relative, uniformly, and no bounded
        # preroll removes it). Rather than silently move every existing caller's output,
        # EasyLoader keeps the decode-and-trim path whenever it has to resample. This test
        # is what says so: the result must stay bit-identical to a full read.
        self.sliceIsExact(join('recorded', 'musicbox.wav'), 22050, [(1.5, 3.5), (5., 6.)])
        self.sliceIsExact(join('recorded', 'guitar_triads.flac'), 44100, [(1.5, 3.5), (5., 6.)])

    def testSeekReplayGain(self):
        # the gain is applied after the slice, so seeking must not disturb it either
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole = self.readSlice(filename, 44100, 0., 1e6, replayGain=-12.)
        found = self.readSlice(filename, 44100, 2., 4., replayGain=-12.)
        self.assertEqualVector(found, self.slice(whole, 44100, 2., 4.))

    def testSeekBoundaries(self):
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole = self.readSlice(filename, 44100, 0., 1e6)
        duration = len(whole)/44100.

        # startTime 0 and no endTime is the default: the whole file
        self.assertEqual(len(whole), 2003649)

        # an endTime past the end of the file stops at the end of the file
        found = self.readSlice(filename, 44100, duration - 1., duration + 100.)
        self.assertEqualVector(found, self.slice(whole, 44100, duration - 1., 1e6))

        # an endTime landing exactly on it reads everything
        self.assertEqual(len(self.readSlice(filename, 44100, 0., duration)), len(whole))

        # a startTime past the end of the file is empty, not an error
        self.assertEqual(len(self.readSlice(filename, 44100, duration + 10., duration + 20.)), 0)

        # and so is an empty slice
        self.assertEqual(len(self.readSlice(filename, 44100, 0., 0.)), 0)
        self.assertEqual(len(self.readSlice(filename, 44100, 3., 3.)), 0)

    def testSeekShortFile(self):
        # shorter than the seek preroll, so the seek lands at the start of the stream
        self.sliceIsExact(join('recorded', 'vignesh.wav'), 44100,
                          [(0.1, 0.2), (2., 3.), (0.25, 1e6)])

    def testSeekStandard(self):
        from essentia.standard import EasyLoader as stdEasyLoader
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole = numpy.array(stdEasyLoader(filename=filename)())
        for startTime, endTime in [(1.5, 3.5), (12.345, 14.345)]:
            found = numpy.array(stdEasyLoader(filename=filename, startTime=startTime,
                                              endTime=endTime)())
            expected = self.slice(whole, 44100, startTime, endTime)
            self.assertEqual(len(found), len(expected))
            self.assert_(numpy.array_equal(found, expected))

    def testInvalidParam(self):
        filename = join(testdata.audio_dir, 'generated','synthesised','impulse','resample',
                        'impulses_1samp_44100.wav')
        self.assertConfigureFails(EasyLoader(), {'filename':'unknown.wav'})
        self.assertConfigureFails(EasyLoader(), {'filename':filename, 'downmix' : 'stereo'})
        self.assertConfigureFails(EasyLoader(), {'filename':filename, 'sampleRate' : 0})
        self.assertConfigureFails(EasyLoader(), {'filename':filename, 'startTime' : -1})
        self.assertConfigureFails(EasyLoader(), {'filename':filename, 'endTime' : -1})
        self.assertConfigureFails(EasyLoader(), {'filename':filename, 'startTime':10, 'endTime' : 1})

    def testResetStandard(self):
        from essentia.standard import EasyLoader as stdEasyLoader
        audiofile = join(testdata.audio_dir,'recorded','musicbox.wav')
        loader = stdEasyLoader(filename=audiofile, startTime=0, endTime=70)
        audio1 = loader()
        audio2 = loader()
        loader.reset()
        audio3 = loader()
        self.assertAlmostEqualVector(audio3, audio1)
        self.assertEqualVector(audio2, audio3)

    def testLoadMultiple(self):
        from essentia.standard import EasyLoader as stdEasyLoader
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = stdEasyLoader(filename=filename)
        audio1 = algo()
        audio2 = algo()
        audio3 = algo()
        self.assertEqual(len(audio1), 441000)
        self.assertEqual(len(audio2), 441000)
        self.assertEqual(len(audio3), 441000)
        self.assertEqualVector(audio2, audio1)
        self.assertEqualVector(audio2, audio3)



suite = allTests(TestEasyLoader_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
