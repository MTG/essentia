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

def cvec(l):
    return numpy.array(l, dtype='c8')

class TestIFFT(TestCase):

    def testDC(self):
        # input is [1, 0, 0, ...] which corresponds to an IFFT of constant magnitude 1
        signalSize = 512
        fftSize = signalSize/2 + 1

        signalDC = zeros(signalSize)
        signalDC[0] = 1.0

        fftInput = [ 1+0j ] * fftSize

        self.assertAlmostEqualVector(signalDC*signalSize, IFFT()(cvec(fftInput)))


    def testNyquist(self):
        # input is [1, -1, 1, -1, ...] which corresponds to a sine of frequency Fs/2
        signalSize = 1024
        fftSize = signalSize/2 + 1

        signalNyquist = ones(signalSize)
        for i in range(signalSize):
            if i % 2 == 1:
                signalNyquist[i] = -1.0

        fftInput = [ 0+0j ] * fftSize
        fftInput[-1] = (1+0j) * signalSize

        self.assertAlmostEqualVector(IFFT()(cvec(fftInput)), signalNyquist*signalSize)

    def testZero(self):
        self.assertAlmostEqualVector(zeros(2048), IFFT()(numpy.zeros(1025, dtype='c8')))


    def testEmpty(self):
        self.assertComputeFails(IFFT(), cvec([]))


    def testRegression(self):
        signal = numpy.sin(numpy.arange(1024, dtype='f4')/1024. * 441 * 2*math.pi)
        self.assertAlmostEqualVector(signal*len(signal), IFFT()(FFT()(signal)), 1e-2)





suite = allTests(TestIFFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

