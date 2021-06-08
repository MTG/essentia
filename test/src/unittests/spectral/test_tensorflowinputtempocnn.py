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
import numpy as np


class TestTensorflowInputTempoCNN(TestCase):

    def testZeroSpectrum(self):
        # Inputting zeros should return zero.
        size = 1024
        self.assertEqualVector(TensorflowInputTempoCNN()(zeros(size)), zeros(40))

    def testRegression(self):
        # Hardcoded analysis parameters
        sampleRate = 11025
        frameSize = 1024
        hopSize = 512

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'),
                           sampleRate=sampleRate)()

        expected = [0.00473826, 0.08461753, 0.4031311,  0.11013637, 0.07605512, 0.20765066,
                    0.18525994, 0.21854699, 0.34972662, 0.4122374,  0.14164223, 0.08753049,
                    0.07250968, 0.02758735, 0.03375721, 0.04653678, 0.0367885,  0.03222116,
                    0.06601233, 0.04644121, 0.06116791, 0.06980634, 0.0918903,  0.13866659,
                    0.07853713, 0.04365063, 0.02569871, 0.0125964,  0.01021744, 0.01130397,
                    0.02596227, 0.03110815, 0.02377434, 0.0178348,  0.01732456, 0.02468131,
                    0.02664435, 0.00746825, 0.00207653, 0.002497]

        tfmf = TensorflowInputTempoCNN()
        frames = [tfmf(frame) for frame in FrameGenerator(audio,
                                                          frameSize=frameSize,
                                                          hopSize=hopSize,
                                                          startFromZero=True)]
        obtained = numpy.mean(array(frames), axis=0)

        self.assertAlmostEqualVector(obtained, expected, 1e-1)

    def testInvalidInput(self):
        self.assertComputeFails(TensorflowInputTempoCNN(), [])

    def testWrongInputSize(self):
        # mel bands should fail for input size different to 1024
        self.assertComputeFails(TensorflowInputTempoCNN(), [0.5] * 1)
        self.assertComputeFails(TensorflowInputTempoCNN(), [0.5] * 514)


suite = allTests(TestTensorflowInputTempoCNN)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
