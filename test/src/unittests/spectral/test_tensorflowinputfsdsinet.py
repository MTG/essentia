#!/usr/bin/env python

# Copyright (C) 2006-2023  Music Technology Group - Universitat Pompeu Fabra
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


class TestTensorflowInputFSDSINet(TestCase):

    # mel-spectrogram analysis parameters
    sr = 22050
    frame_size = 660
    hop_size = 220
    n_bands = 96

    def melspectrogram(self, signal):
        """Compute the mel-spectrogram of a signal"""

        algo = TensorflowInputFSDSINet()
        frames = [algo(frame) for frame in FrameGenerator(
            signal,
            frameSize=self.frame_size,
            hopSize=self.hop_size,
        )]

        return numpy.array(frames)

    def testZeroSpectrum(self):
        """Test inputting a vector of zeros

        Inputting zeros should return the cutoff value from UnaryOperator (i.e., 1e-30).
        https://essentia.upf.edu/reference/std_UnaryOperator.html
        """

        log_cutoff = -30

        self.assertEqualVector(TensorflowInputFSDSINet()(zeros(self.frame_size)), log_cutoff * ones(self.n_bands))

    def testRegressionDC(self):
        """Regression of the mel-spectrogram for a continuous (DC) signal"""

        expected_file = join(
            filedir(),
            "tensorflowinputfsdsinet",
            "dc_melspectrogram_fsdsinet.npy"
        )
        expected = numpy.load(expected_file)

        dc = numpy.ones(self.sr, dtype="float32")
        found = self.melspectrogram(dc)

        # Essentia generates an additional frame because of its zero-padding strategy.
        # We can discard it for testing purposes.
        found = found[:expected.shape[0], :]

        # The high relative error w.r.t. the original implementation comes mainly from using
        # floats instead of double precision. Two points where the precision plays an important
        # role are:
        # - 1. The FFT. If the signal is very tonal (or DC), the energy (and the truncation
        #  error) accumulates at certain FFT bins.
        # - 2. The mel-spectrogram frequency calculation. Some bins can suffer deviations in the
        #  order of the hundreds of Hz.
        #
        # While we acknowledge this deviation, it does not have a significant impact in our target
        # applications (i.e., inference with neural networks). - Pablo A.
        self.assertAlmostEqualMatrix(found, expected, 1e0)

    def testRegressionVoiceSignal(self):
        """Regression of the mel-spectrogram for a voice signal"""

        audio = MonoLoader(
            filename=join(testdata.audio_dir, 'recorded/vignesh.wav'),
            sampleRate=self.sr,
        )()

        expected_file = join(
            filedir(),
            "tensorflowinputfsdsinet",
            "vignesh_melspectrogram_fsdsinet.npy"
        )
        expected = numpy.load(expected_file)

        found = self.melspectrogram(audio)
        found = found[:expected.shape[0], :]

        # Check comment on testRegressionDC.
        self.assertAlmostEqualMatrix(found, expected, 1e0)

    def testInvalidInput(self):
        """Test an empty input"""

        self.assertComputeFails(TensorflowInputFSDSINet(), [])

    def testWrongInputSize(self):
        """Test wrong input sizes

        The algorithm should fail for input size different to 660.
        """

        self.assertComputeFails(TensorflowInputFSDSINet(), [0.5] * 1)
        self.assertComputeFails(TensorflowInputFSDSINet(), [0.5] * 514)


suite = allTests(TestTensorflowInputFSDSINet)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
