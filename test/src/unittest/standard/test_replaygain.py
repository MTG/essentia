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

#testdir = join(filedir(), 'replaygain')

class TestReplayGain(TestCase):

    def testZero(self):
        sampleRate = 44100
        input = [0.0] * int(sampleRate * 1)
        replayGainDiff = ReplayGain(sampleRate=sampleRate)(input)
        dbSilence = -90 # by definition in essentiamath
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
