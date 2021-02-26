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
from essentia.streaming import MonoLoader as sMonoLoader
from essentia.streaming import RhythmExtractor as sRhythmExtractor


class TestRhythmTransform(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(RhythmTransform(), {'frameSize': -1})
        self.assertConfigureFails(RhythmTransform(), {'hopSize': -1})
    """
    FIXME: Work in progress for Regression Test
    """
    def testRegression(self):
        # Simple regression test, comparing normal behaviour
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded', 'techno_loop.wav'),sampleRate = 44100)()
        sampleRate   = 44100
        frameSize    = 8192
        hopSize      = 1024
        rmsFrameSize = 256
        rmsHopSize   = 32

        w = Windowing(type='blackmanharris62')
        spectrum = Spectrum()
        melbands = MelBands(sampleRate=sampleRate, numberBands=40, lowFrequencyBound=0, highFrequencyBound=sampleRate/2)
        # Maybe test this in streaming mode
        for frame in FrameGenerator(audio=audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
            bands = melbands(spectrum(w(frame)))
            #pool.add('melbands', bands)
        

        rt = RhythmExtractor(frameSize=rmsFrameSize, hopSize=rmsHopSize)(bands)

        print(rt)
        #FIXME forcing a fail to indocate Test Case is work in progress
        self.assertEqual(0, 1)

        #rt_mean = numpy.mean(rt, axis=0)
        #bin_resoluion = 5.007721656976744
        #print("Estimated BPM: %0.1f" % float(numpy.argmax(rt_mean) * bin_resoluion))

    def testZero(self):
        # FIXME  Make array dims more realistic
        # FIXME  sticking with defaults for now
        bands = [zeros(256), zeros(256)]
        rt = RhythmTransform()(bands)
        self.assertEqual(len(rt), 1)

        bands = [zeros(512), zeros(512), zeros(512)]
        rt = RhythmTransform()(bands)
        self.assertEqual(len(rt), 1)

    def testConstantInput(self):
        # FIXME  Make array dims more realistic
        # FIXME  sticking with defaults for now        
        bands = [ones(256), ones(256)]
        rt = RhythmTransform()(bands)
        self.assertEqual(len(rt), 1)

        bands = [ones(512), ones(512), ones(512)]
        rt = RhythmTransform()(bands)
        self.assertEqual(len(rt), 1)



    def testEmpty(self):
        # FIXME  Empty bands causes Segmentation fault (core dumped)
        bands = []
        rt = RhythmTransform()(bands) 
        self.assertEqualVector(rt, [])


suite = allTests(TestRhythmTransform)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
