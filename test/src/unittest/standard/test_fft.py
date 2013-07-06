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
import math


class TestFFT(TestCase):

    def testDC(self):
        # input is [1, 0, 0, ...] which corresponds to an FFT of constant magnitude 1
        signalSize = 512
        fftSize = signalSize/2 + 1

        signalDC = zeros(signalSize)
        signalDC[0] = 1.0

        expected = [ 1+0j ] * fftSize

        self.assertAlmostEqualVector(FFT()(signalDC), expected)

    def testNyquist(self):
        # input is [1, -1, 1, -1, ...] which corresponds to a sine of frequency Fs/2
        signalSize = 1024
        fftSize = signalSize/2 + 1

        inputNyquist = ones(signalSize)
        for i in range(signalSize):
            if i % 2 == 1:
                inputNyquist[i] = -1.0

        expected = [ 0+0j ] * fftSize
        expected[-1] = (1+0j) * signalSize

        self.assertAlmostEqualVector(FFT()(inputNyquist), expected)

    def testZero(self):
        self.assertEqualVector(FFT()(zeros(2048)), zeros(1025))



    def testWeirdSizes(self):
        self.assertComputeFails(FFT(), [])

        # values from Matlab/Octave
        self.assertComputeFails(FFT(), [1])
        self.assertComputeFails(FFT(), [1, 0.5, 0.2])

        # These tests should be reactivated when the FFT will work on odd-sized arrays
        '''
        self.assertAlmostEqualVector(FFT()([1]), [1])

        self.assertAlmostEqualVector(FFT()([1, 0.5, 0.2]),
                                     [ 1.7+0j, 0.65-0.2598076211j ])

        self.assertAlmostEqualVector(FFT()([1, 4, 2, 5, 3]),
                                     [ 15+0j, -2.5+0.8122992405j, -2.5-3.4409548011j ])
        '''


    def testRegression(self):
        inputSignal = numpy.sin(numpy.arange(1024, dtype='f4')/1024. * 441 * 2*math.pi)
        expected = readComplexVector(join(filedir(), 'fft', 'fft_output.txt'))
        expected = expected[:len(inputSignal)/2+1]

        # readjust to our precision, otherwise 0.001 compared to 1e-12 would
        # give a 1e9 difference...
        for i, value in enumerate(expected):
            if abs(value) < 1e-7:
                expected[i] = 0+0j

        self.assertAlmostEqualVector(FFT()(inputSignal), expected, 1e-2)






suite = allTests(TestFFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

