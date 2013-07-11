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
        self.load(44100, 22050, filename, "left" , 0., 0., 10.);
        self.load(44100, 48000, filename, "right", -15., 3.34, 5.68);
        self.load(44100, 11025, filename, "mix"  , 30., 0.168, 8.32);

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
        audio1 = loader();
        audio2 = loader();
        loader.reset();
        audio3 = loader();
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
        self.assertEquals(len(audio1), 441000);
        self.assertEquals(len(audio2), 441000);
        self.assertEquals(len(audio3), 441000);
        self.assertEqualVector(audio2, audio1)
        self.assertEqualVector(audio2, audio3)



suite = allTests(TestEasyLoader_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
