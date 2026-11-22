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
from essentia.streaming import EasyLoader, EqloudLoader
import sys
class TestEqloudLoader_Streaming(TestCase):

    def load(self, inputSampleRate, outputSampleRate,
                   eqloudfilename, normalfilename,
                   downmix, replayGain, startTime, endTime):

        eqloudloader = EqloudLoader(filename=normalfilename,
                                    sampleRate = outputSampleRate,
                                    downmix = downmix,
                                    startTime = startTime,
                                    endTime = endTime,
                                    replayGain = replayGain)

        easyloader = EasyLoader(filename=eqloudfilename,
                                sampleRate = outputSampleRate,
                                downmix = downmix,
                                startTime = startTime,
                                endTime = endTime,
                                replayGain = replayGain)
        pool = Pool()

        easyloader.audio >> (pool, 'expected')
        run(easyloader)

        eqloudloader.audio >> (pool, 'eqloud')
        run(eqloudloader)

        for val1, val2 in zip(pool['eqloud'][outputSampleRate:],
                              pool['expected'][outputSampleRate:]):
              self.assertAlmostEqual(val1-val2, 0, 5e-3)


    def testNoResample(self):
        eqloud=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds_eqloud.wav')
        normal=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds.wav')
        self.load(44100, 44100, eqloud, normal, "left" , -6.0, 0., 30.)
        self.load(44100, 44100, eqloud, normal, "left", -6.0, 3.35, 5.68)
        self.load(44100, 44100, eqloud, normal, "left"  , -6.0, 0.169, 8.333)

    def testResample(self):
        eqloud=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds_eqloud.wav')
        normal=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds.wav')
        self.load(44100, 48000, eqloud, normal, "left", -6.0, 3.35, 5.68)
        self.load(44100, 32000, eqloud, normal, "left", -6.0, 3.35, 5.68)



    # ------------------------------------------------------------------------------------
    # EqloudLoader gained the same seeking as EasyLoader (issue #771), and the same
    # obligation: the slice it returns must be the slice a full read would have given.

    def readSlice(self, filename, sampleRate, startTime, endTime):
        loader = EqloudLoader(filename=filename, sampleRate=sampleRate, downmix='mix',
                              startTime=startTime, endTime=endTime, replayGain=-6.)
        pool = Pool()
        loader.audio >> (pool, 'audio')
        run(loader)
        if 'audio' not in pool.descriptorNames():
            return numpy.array([], dtype='float32')
        return numpy.array(pool['audio'])

    def sliceIsExact(self, name, sampleRate, spans):
        # The reference cannot be a slice of a full EqloudLoader read: the equal-loudness
        # filter is an IIR whose state depends on everything it has already seen, so a slice
        # of a whole-file read has never equalled a read of that slice, patch or no patch.
        # EqloudLoader IS EasyLoader followed by EqualLoudness, so compose the reference
        # that way instead -- exact, and independent of where the loader started.
        from essentia.standard import EqualLoudness
        filename = join(testdata.audio_dir, name)
        for startTime, endTime in spans:
            found = self.readSlice(filename, sampleRate, startTime, endTime)
            easy = EasyLoader(filename=filename, sampleRate=sampleRate, downmix='mix',
                              startTime=startTime, endTime=endTime, replayGain=-6.)
            pool = Pool()
            easy.audio >> (pool, 'audio')
            run(easy)
            expected = EqualLoudness(sampleRate=sampleRate)(pool['audio'])
            self.assertEqual(len(found), len(expected))
            self.assertAlmostEqualVector(found, expected, 1e-6)

    def testSeek(self):
        self.sliceIsExact(join('recorded', 'musicbox.wav'), 44100,
                          [(0., 2.), (1.5, 3.5), (12.345, 14.345)])
        self.sliceIsExact(join('recorded', 'techno_loop.mp3'), 44100,
                          [(1.5, 3.5), (25., 30.)])

    def testSeekResampledIsUnchanged(self):
        # see EasyLoader: a resampled slice deliberately keeps the decode-and-trim path
        self.sliceIsExact(join('recorded', 'musicbox.wav'), 32000, [(1.5, 3.5), (5., 6.)])

    def testSeekBoundaries(self):
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        whole = self.readSlice(filename, 44100, 0., 1e6)
        duration = len(whole)/44100.
        self.assertEqual(len(self.readSlice(filename, 44100, 0., duration)), len(whole))
        self.assertEqual(len(self.readSlice(filename, 44100, duration + 10., duration + 20.)), 0)
        self.assertEqual(len(self.readSlice(filename, 44100, 3., 3.)), 0)

    def testInvalidParam(self):
        filename = join(testdata.audio_dir, 'generated','synthesised','impulse','resample',
                        'impulses_1samp_44100.wav')
        self.assertConfigureFails(EqloudLoader(), {'filename':'unknown.wav'})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'downmix' : 'stereo'})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'sampleRate' : 0})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'startTime' : -1})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'endTime' : -1})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'startTime':10, 'endTime' : 1})

    def testResetStandard(self):
        from essentia.standard import EqloudLoader as stdEqloudLoader
        audiofile = join(testdata.audio_dir,'recorded','musicbox.wav')
        loader = stdEqloudLoader(filename=audiofile, endTime=31)
        audio1 = loader()
        audio2 = loader()
        loader.reset()
        audio3 = loader()
        self.assertAlmostEqualVector(audio3, audio1)
        self.assertEqualVector(audio2, audio1)

    def testLoadMultiple(self):
        from essentia.standard import EqloudLoader as stdEqloudLoader
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = stdEqloudLoader(filename=filename)
        audio1 = algo()
        audio2 = algo()
        audio3 = algo()
        self.assertEqual(len(audio1), 441000)
        self.assertEqual(len(audio2), 441000)
        self.assertEqual(len(audio3), 441000)
        self.assertEqualVector(audio2, audio1)
        self.assertEqualVector(audio2, audio3)



suite = allTests(TestEqloudLoader_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
