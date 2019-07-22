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
import math


class TestFFT(TestCase):

    def testDC(self):
        # input is [1, 0, 0, ...] which corresponds to an FFT of constant magnitude 1
        signalSize = 512
        fftSize = int(signalSize/2) + 1

        signalDC = zeros(signalSize).astype(numpy.complex64)
        signalDC[0] = 1.0 + 0j

        expected = [ 1+0j ] * fftSize

        self.assertAlmostEqualVector(FFTC()(signalDC), expected)

    def testNyquist(self):
        # Input is [1, -1, 1, -1, ...] which corresponds to a sine of frequency Fs/2.
        signalSize = 1024
        fftSize = int(signalSize/2) + 1

        inputNyquist = ones(signalSize).astype(numpy.complex64)
        for i in range(signalSize):
            if i % 2 == 1:
                inputNyquist[i] = -(1.0 + 0j)

        expected = [ 0+0j ] * fftSize
        expected[-1] = (1+0j) * signalSize

        self.assertAlmostEqualVector(FFTC()(inputNyquist), expected)

    def testZero(self):
        self.assertEqualVector(FFTC()(zeros(2048).astype(numpy.complex64)),
                                      zeros(1025).astype(numpy.complex64))

    def testWeirdSizes(self):
        self.assertComputeFails(FFTC(), array([]).astype(numpy.complex64))

        # Values from Matlab/Octave.
        self.assertComputeFails(FFTC(), array([1]).astype(numpy.complex64))
        self.assertComputeFails(FFTC(), array([1, 0.5, 0.2]).astype(numpy.complex64))

    def testRegression(self):
        size = 2048
        test_signal = array(numpy.random.rand(2, size))
        test_signal = test_signal[0] + 1j * test_signal[1]

        # Use random values and compare with the Numpy Complex FFT.
        self.assertAlmostEqualVector(FFTC(negativeFrequencies=True)(test_signal), 
                                     numpy.fft.fft(test_signal), 1e-4)



suite = allTests(TestFFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
