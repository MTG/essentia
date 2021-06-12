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
import statistics


class TestMedianFilter(TestCase):

    def testInputTooSmall(self):
        original_signal = ones(10)
        self.assertRaises(RuntimeError, lambda: MedianFilter(kernelSize=11)(original_signal))
    
    def testInputEqKernalSize(self):
        original_signal = ones(11)
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

    def testRangeChecks(self):
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
   
      
    def testFilterPadding(self):
        # Use Random input data and check that the beginning and the end 
        # are padded with the first and last values in the input vector.    
        x1 = [12, 2, 5, 6, 51, 45646, 614]         
        x2 = [6, 51, 45646, 614, 1, 14, 6 ,7]         
        x3 = [16, 51, 45, 41, 45, 51, 45, 4, 51, 45, 6, 46, 3]         

        y1= std.MedianFilter(kernelSize = 3)(x1)
        y2= std.MedianFilter(kernelSize = 5)(x2)
        y3= std.MedianFilter(kernelSize = 7)(x3)
        self.assertEqual(y1[0], x1[0])
        self.assertEqual(y1[len(x1)-1], x1[len(x1)-1])        
        self.assertEqual(y2[0], x2[0])
        self.assertEqual(y2[len(x2)-1], x2[len(x2)-1])            
        self.assertEqual(y3[0], x3[0])        
        self.assertEqual(y3[len(x3)-1], x3[len(x3)-1])                

    def testRegressionMedianPart(self):
        # Manually pass a median filter through the x array below
        x = [-2, 2, 13, 16, 9, 14, 4]         
 
        # Use library median calculation on 4 shifted positions
        y1 = statistics.median(x[0:3])
        y2 = statistics.median(x[1:4])
        y3 = statistics.median(x[2:5])
        y4 = statistics.median(x[3:6])        
        calculated_median= [ y1, y2, y3, y4]
        
        # Test with kernel = 3
        y= std.MedianFilter(kernelSize = 3)(x)

        # Clip off the padded parts and check against above values
        y_pads_removed = y[1:len(y)-2]
        self.assertEqualVector(calculated_median, y_pads_removed)

suite = allTests(TestMedianFilter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

