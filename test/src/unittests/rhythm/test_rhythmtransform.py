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
import numpy as np

listZeros=[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
  0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
  0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
  0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
  0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
  0.,0.,0.,0.,0.,0.,0.,0.,0.]]

class TestRhythmTransform(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(RhythmTransform(), {'frameSize': -1})
        self.assertConfigureFails(RhythmTransform(), {'hopSize': -1})

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

        pool = Pool()

        # Maybe test this in streaming mode
        for frame in FrameGenerator(audio=audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
            bands = melbands(spectrum(w(frame)))
            pool.add('melbands', bands)

        #print("Mel band frames: %d" % len(pool['melbands']))
        #print("Rhythm transform frames: %d" % int(len(pool['melbands']) / 32))

        rhythmtransform = RhythmTransform(frameSize=rmsFrameSize, hopSize=rmsHopSize)
        rt = rhythmtransform(pool['melbands'])

        # This code stores reference values in a file for later loading.
        # np.save('rhythmtransform.npy', rt)
        # Reference samples are loaded as expected values
        expected_rhythmtransform_npy = np.load(join(filedir(), 'rhythmtransform/rhythmtransform.npy'))
        expected_rhythmtransform = expected_rhythmtransform_npy.tolist()

        self.assertAlmostEqualVectorFixedPrecision(expected_rhythmtransform[0], rt[0] ,2)
        self.assertAlmostEqualVectorFixedPrecision(expected_rhythmtransform[1], rt[1] ,2)

    def testAllEmpty(self):
        # Testing for zero number of input mel-frames
        bandsAllEmpty = []
        self.assertRaises(EssentiaException, lambda: RhythmTransform()(bandsAllEmpty))
     
    def testMultipleEmptySlots(self):
        # Testing for non-zero number of empty slots
        multipleEmptySlots = [[],[]]
        self.assertRaises(EssentiaException, lambda: RhythmTransform()(multipleEmptySlots))

    def testNonUniformEmptyInput(self):
        nEmptySlots = [[],[0,0]]
        self.assertRaises(EssentiaException, lambda: RhythmTransform()(nEmptySlots))

    def testZeros1Darray(self):
        bands = [zeros(1024)]
        rt = RhythmTransform()(bands)
        rt_list = rt.tolist()
        self.assertEqualVector(listZeros,rt_list)

    def testZeros2Darray(self):
        # non-zero number of input frames, but each mel-frame vector is empty
        bands = [zeros(1024),zeros(1024)]
        rt = RhythmTransform()(bands)
        rt_list = rt.tolist()
        self.assertEqualVector(listZeros,rt_list)

    def testConstantInput1Darray(self):
        bands = [ones(1024)]
        rt = RhythmTransform()(bands)
        rt_list = rt.tolist()
        self.assertEqualVector(listZeros,rt_list)

    def testConstantInput2Darray(self):
        bands = [ones(1024),ones(1024)]
        rt = RhythmTransform()(bands)
        rt_list = rt.tolist()
        self.assertEqualVector(listZeros,rt_list)

    def testNonUniformInput(self):
        bands = [ones(512), ones(1024)]
        self.assertRaises(EssentiaException, lambda: RhythmTransform()(bands))

suite = allTests(TestRhythmTransform)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
