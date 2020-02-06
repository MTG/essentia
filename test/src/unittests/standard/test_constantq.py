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


class TestConstantQ(TestCase):

    def testRegression(self):
        expected = numpy.load(join(filedir(), 'constantq/constantq_values.npy'))

        frame_size = 32768
        hop_size = frame_size // 4

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'))()

        w = Windowing()
        cqt = ConstantQ()

        predicted = numpy.array([cqt(w(frame)) for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)])

        self.assertAlmostEqualVector(numpy.mean(predicted, axis=0), expected, 1e-7)

    def testRegressionNoZeroPhase(self):
        expected = numpy.load(join(filedir(), 'constantq/constantq_values.npy'))

        frame_size = 32768
        hop_size = frame_size // 4
        zeroPhase=False

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'))()

        w = Windowing(zeroPhase=zeroPhase)
        cqt = ConstantQ(zeroPhase=zeroPhase)

        predicted = numpy.array([cqt(w(frame)) for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)])

        self.assertAlmostEqualVector(numpy.mean(predicted, axis=0), expected, 1e-7)

    def testZero(self):
        inputSize = 2**15
        signalZero = [0] * inputSize
        output = numpy.mean(numpy.abs(ConstantQ()(signalZero)))
        self.assertEqual(0, output)

    def testEmpty(self):
        # Checks whether an empty input vector yields an exception
        self.assertComputeFails(ConstantQ(), [])

    def testOne(self,):
        # Checks for a single value
        self.assertComputeFails(ConstantQ(), [1])

    def testInvalidParam(self):
        self.assertConfigureFails(ConstantQ(), {'minFrequency': 30000})  # Min bin above Nyquist
        self.assertConfigureFails(ConstantQ(), {'numberBins': 0})  # No CQ bins
        self.assertConfigureFails(ConstantQ(), {'binsPerOctave': 0})  # No CQ bins
        self.assertConfigureFails(ConstantQ(), {'sampleRate': 400}) # With this sample rate the kernels are out of range
        self.assertConfigureFails(ConstantQ(), {'numberBins': 10 * 12})  # Max bin above Nyquist
        self.assertConfigureFails(ConstantQ(), {'minimumKernelSize': 1})  # FFT can not be done
        self.assertConfigureFails(ConstantQ(), {'scale': 0})  # Kernels of size 0


suite = allTests(TestConstantQ)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
