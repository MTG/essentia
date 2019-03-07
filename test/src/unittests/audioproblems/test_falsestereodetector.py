#!/usr/bin/env python

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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


import numpy as np
from math import *

from essentia_test import *


class TestFalseStereoDetector(TestCase):
    def testZero(self):
        size = 512
        zeros = array([(0., 0.)] * size)
        self.assertEqual(FalseStereoDetector()(zeros),
                         (0, 0.))

    def testOneChannelBias(self):
        size = 512
        zeros = array([(1., 0.)] * size)
        self.assertEqual(FalseStereoDetector()(zeros),
                         (0, 0.))

    def testInvalidParam(self):
        self.assertConfigureFails(FalseStereoDetector(), {'silenceThreshold': 0})
        self.assertConfigureFails(FalseStereoDetector(), {'correlationThreshold': 1.1})

    def testRealSignalDuplicatedChannels(self,):
        monoAudio = MonoLoader(filename=join(testdata.audio_dir,
                                             'recorded/vignesh.wav'))()

        stereoAudio = StereoMuxer()(monoAudio, monoAudio)

        # If both channels are equal the flag 'isFalseStereo' should be
        # activated and the value of the correlation should be 1.
        self.assertEqual(FalseStereoDetector()(stereoAudio),
                         (1, 1.))

    def testRealSignalOutOfOPhase(self,):
        monoAudio = MonoLoader(filename=join(testdata.audio_dir,
                                             'recorded/vignesh.wav'))()

        stereoAudio = StereoMuxer()(monoAudio, -monoAudio)

        # If the signals are the same but out of phase, the correlation should
        # be -1, but we are not activating the flag in this case.
        self.assertEqual(FalseStereoDetector()(stereoAudio),
                         (0, -1.))

    def testRegression(self,):
        stereoAudio = AudioLoader(filename=join(testdata.audio_dir,
                                                'recorded/dubstep.wav'))()[0]
        # We are not interested in the value of the correlation here.
        # We are happy if the flag 'isfalseStereo' is not triggered.
        self.assertEqual(FalseStereoDetector()(stereoAudio)[0], 0)

    def testScaleAndShift(self):
        size = 512

        # We use a synthetic to evaluate robustness against
        # scale and shift.
        signal = array([(1., 2.), (-1., 0.)] * size)
        self.assertEqual(FalseStereoDetector()(signal),
                         (1, 1.))

    def testInverseScaleAndShift(self):
        size = 512

        # Same test with inverse phase.
        signal = array([(1., -2.), (-1., 0.)] * size)
        self.assertEqual(FalseStereoDetector()(signal),
                         (0, -1.))

    def testSilenceThreshold(self):
        size = 512
        s = db2pow(-70)
        # Signals are correlated but scaled to be under the silence threshold.
        # In this case, the algorithm should return 0 as any correlation can be
        # originated from noise or dithering and we cannot extract conclusions
        # about the actual music signal.
        signal = array([(1. * s, 2. * s), (-1. * s, 0. * s)] * size)
        self.assertEqual(FalseStereoDetector()(signal),
                         (0, 0.))

suite = allTests(TestFalseStereoDetector)


if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
