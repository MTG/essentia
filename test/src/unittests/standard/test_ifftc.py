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

def cvec(l):
    return numpy.array(l, dtype='c8')

class TestIFFT(TestCase):

    def testDC(self):
        # input is [1, 0, 0, ...] which corresponds to an IFFT of constant magnitude 1
        signalSize = 512

        freqSignal = zeros(signalSize).astype(numpy.complex64)
        freqSignal[0] = (1. + 0j) * signalSize

        timeSignal = [1.+ 0j] * signalSize

        self.assertAlmostEqualVector(cvec(timeSignal), IFFTC()(freqSignal))


    def testNyquist(self):
        # input is [1, -1, 1, -1, ...] which corresponds to a sine of frequency Fs/2
        signalSize = 1024

        timeSignal = ones(signalSize).astype(numpy.complex64)
        for i in range(signalSize):
            if i % 2 == 1:
                timeSignal[i] = -1.0

        freqSignal = [ 0+0j ] * signalSize
        freqSignal[signalSize // 2] = (1+0j) * signalSize

        self.assertAlmostEqualVector(IFFTC()(cvec(freqSignal)), timeSignal)

    def testZero(self):
        self.assertAlmostEqualVector(zeros(2048), IFFT()(numpy.zeros(1025, dtype='c8')))


    def testEmpty(self):
        self.assertComputeFails(IFFT(), cvec([]))

    def testRegression(self):
        signal = (numpy.sin(numpy.arange(1024, dtype='f4')/1024. * 441 * 2 * math.pi) +\
                 1j * numpy.cos(numpy.arange(1024, dtype='f4')/1024. * 441 * 2 * math.pi)).astype(numpy.complex64)
        self.assertAlmostEqualVector(signal, IFFTC()(FFTC(negativeFrequencies=True)(signal)), 1e-2)



suite = allTests(TestIFFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

