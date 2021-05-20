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
from math import *
from numpy import *

from essentia import *
from math import *
import essentia.standard as std


class TestMedianFilter(TestCase):

    def testInputTooSmall(self):
        original_signal = ones(10)
        self.assertRaises(RuntimeError, lambda: MedianFilter(kernelSize=11)(original_signal))

    def testOddKernelSize(self):
        original_signal = ones(256)
        self.assertRaises(RuntimeError, lambda: MedianFilter(kernelSize=10)(original_signal))
    
    def testZeros(self):
        original_signal = zeros(256)
        medianfiltered_signal = MedianFilter()(original_signal)
        self.assertEqualVector(medianfiltered_signal, original_signal)

    def testEmpty(self):
        original_signal = []
        self.assertRaises(RuntimeError, lambda: MedianFilter()(original_signal))

    def testRegressionRange(self):
        sr = 44100.
        pi2 = 2*pi
        signal = [.25*cos(t*pi2*5/sr) + \
                  .25*cos(t*pi2*50/sr) + \
                  .25*cos(t*pi2*500./sr) + \
                  .25*cos(t*pi2*5000./sr)
                  for t in range(44100)]

        filteredSignal = MedianFilter(kernelSize=11)(signal)

        s = Spectrum()(signal)
        sf = Spectrum()(filteredSignal)

        for i in range(1000):
            if s[i] > 10:
                self.assertTrue(sf[i] > 0.5*s[i])

        for i in range(1001, len(sf)):
            if s[i] > 10:
                self.assertTrue(sf[i] < 0.5*s[i])
    
    def testRegression(self):
        sr = 44100
        index = 0
        original_signal = [10, 10, 10, 20, 5, 5, 40, 5, 5, 80, 5, 5]
        # Test with kernel = 3
        # The output is "smoothed" (outliers 20,40, 80 removed)
        expected_output = [10., 10., 10., 10.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.]
        medianfiltered_signal = std.MedianFilter(kernelSize = 3)(original_signal)
        self.assertEqualVector(medianfiltered_signal, expected_output)

        # Test with kernel = 5 
        original_signal = [10, 10, 10, 20, 5, 15, 25, 40, 35, 25, 15, 80, 75, 65, 85]
        # You can see the edges are smoothed out
        expected_output = [10, 10, 10, 10, 15, 20, 25, 25, 25, 35, 35, 65, 75, 80, 85] 
        medianfiltered_signal = std.MedianFilter(kernelSize=5)(original_signal)
        self.assertEqualVector(medianfiltered_signal, expected_output)
   
suite = allTests(TestMedianFilter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

