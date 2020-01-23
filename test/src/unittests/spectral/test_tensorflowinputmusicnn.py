#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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


class TestTensorflowInputMusiCNN(TestCase):

    def testZeroSpectrum(self):
        # Inputting zeros should return zero.
        size = 512
        self.assertEqualVector(TensorflowInputMusiCNN()(zeros(size)), zeros(96))

    def testRegression(self):
        # Hardcoded analysis parameters
        sampleRate = 16000
        frameSize = 512
        hopSize = 256

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'),
                           sampleRate=sampleRate)()

        expected = [0.60742337, 0.30574673, 0.45560792, 1.256332,   2.4021673,  3.4365354,
                    3.619476,   3.0021546,  2.1846066,  1.5598421,  1.4810421,  2.2823677,
                    2.679103,   2.86526,    2.6989846,  2.2553382,  2.219071,   2.3255587,
                    2.8502884,  2.9091403,  2.7634032,  2.4637504,  1.8271459,  1.522163,
                    1.7100089,  1.8728845,  1.6959977,  1.3508593,  0.9608341,  0.9133418,
                    1.0304681,  1.493988,   1.6636051,  1.4825928,  1.1171728,  0.93050385,
                    1.2989489,  1.7412357,  1.7828379,  1.5357956,  1.0274258,  1.4541839,
                    1.8527577,  1.8838495,  1.4812496,  1.4385983,  2.1568356,  2.3677773,
                    1.9438239,  1.5913178,  1.8563453,  1.7012404,  1.1431638,  1.0995349,
                    1.1092283,  0.74655735, 0.6698305,  0.7290597,  0.47290954, 0.64479357,
                    0.7136836,  0.9934933,  1.3321629,  1.1683794,  1.2097421,  1.1075293,
                    1.0301174,  0.9288259,  0.8876033,  0.8086145,  0.9854008,  1.0852002,
                    1.2092237,  1.2816739,  1.2066866,  0.52382684, 0.1494276,  0.08070073,
                    0.09443883, 0.12541461, 0.11942478, 0.1558171,  0.17869301, 0.36044103,
                    0.5242918,  0.7467586,  0.8322874,  0.7977463,  0.8188014,  0.80939233,
                    0.74459517, 0.5341967,  0.4339693,  0.33098528, 0.10355855, 0.00549104]

        tfmf = TensorflowInputMusiCNN()
        frames = [tfmf(frame) for frame in FrameGenerator(audio,
                                                          frameSize=frameSize,
                                                          hopSize=hopSize)]
        obtained = numpy.mean(array(frames), axis=0)

        self.assertAlmostEqualVector(obtained, expected, 1e-2)

    def testInvalidInput(self):
        self.assertComputeFails(TensorflowInputMusiCNN(), [])

    def testWrongInputSize(self):
        # mel bands should fail for input size different to 512
        self.assertComputeFails(TensorflowInputMusiCNN(), [0.5] * 1)
        self.assertComputeFails(TensorflowInputMusiCNN(), [0.5] * 514)


suite = allTests(TestTensorflowInputMusiCNN)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
