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
import gc
import weakref

#testdir = join(filedir(), 'replaygain')

class TestReplayGain(TestCase):

    def testInitialization(self):
        rg = ReplayGain()
        self.assertIsInstance(rg, ReplayGain)

    def testDefaultSampleRate(self):
        rg = ReplayGain()
        print (rg.parameter("sampleRate"))
        self.assertEqual(rg.parameter("sampleRate"), 44100)

    def testConfigureViaCompute(self):
        sampleRate = 48000
        input = [0.0] * int(sampleRate * 0.1)

        rg = ReplayGain(sampleRate=sampleRate)
        gain = rg(input)

        self.assertIsInstance(gain, float)

    def testReset(self):
        sampleRate = 44100
        input = [0.0] * int(sampleRate * 0.1)

        rg = ReplayGain(sampleRate=sampleRate)
        first = rg(input)

        rg.reset()
        second = rg(input)

        self.assertAlmostEqual(first, second)

    def testMultipleComputesProduceDifferentResults(self):
        sampleRate = 44100
        rg = ReplayGain(sampleRate=sampleRate)

        input1 = [0.0] * int(sampleRate * 0.1)
        input2 = [0.1] * int(sampleRate * 0.1)

        g1 = rg(input1)
        g2 = rg(input2)

        self.assertNotEqual(g1, g2)

    def testDestructor(self):
        rg = ReplayGain(sampleRate=44100)
        ref = weakref.ref(rg)

        del rg
        gc.collect()
        self.assertIsNone(ref())

    def testZero(self):
        sampleRate = 44100
        input = [0.0] * int(sampleRate * 1)
        replayGainDiff = ReplayGain(sampleRate=sampleRate)(input)
        dbSilence = -100 # by definition in essentiamath
        loudness_ref = -31.492595672607422
        self.assertAlmostEqual(replayGainDiff,loudness_ref - dbSilence)

    def testEmpty(self):
        # Verifies that an exception is thrown when given an empty input
        self.assertComputeFails(ReplayGain(sampleRate=44100), [])

    def testInvalidInput(self):
        # Verifies that an exception is thrown if not enough input is given
        sampleRate = 44100
        inputSize = int(sampleRate * 0.05) # this is the minimum input size (0.05s
        input = [0.0] * (inputSize - 1)

        self.assertComputeFails(ReplayGain(sampleRate=sampleRate), input)

    def testPinkReference(self):
        # Verifies that pink noise returns a 0 difference, since it is the
        # reference that we are using for this algorithm
        sampleRate = 44100
        pinkNoisePath = join(testdata.audio_dir, 'generated', 'synthesised', 'noise_pink.wav')
        input = MonoLoader(filename = pinkNoisePath,
                           sampleRate = sampleRate)()
        replayGainDiff = ReplayGain(sampleRate=sampleRate)(input)
        self.assertAlmostEqual(replayGainDiff, 0.0, 0.1)

suite = allTests(TestReplayGain)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
