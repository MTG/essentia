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

class TestMaxFilter(TestCase):

    def testZeros(self):
      original_signal = zeros(8)
      maxfiltered_signal = MaxFilter()(original_signal)
      self.assertEqualVector(maxfiltered_signal, original_signal)

    def testEmpty(self):
      original_signal = []
      self.assertRaises(RuntimeError, lambda: MaxFilter()(original_signal))
    
    def testRegression(self):
      sr = 44100
      index = 0
      original_signal = [10, 10, 10, 20, 5, 5, 40, 5, 5, 80, 5, 5] 
      
      # Test with filter width = 3
      # The values "5" are filtered out by the max filters
      expected_output = [10, 10, 10, 20, 20, 20, 40, 40, 40, 80, 80, 80]
      maxfiltered_signal = std.MaxFilter(width = 3)(original_signal)
      self.assertEqualVector(maxfiltered_signal, expected_output)

      # Test with filter width = 4
      original_signal = [10, 10, 10, 20, 5, 15, 25, 40, 35, 25, 15, 80, 75, 65, 85]
      # You can see the max values are "held" for every group of "width=4" elements
      expected_output = [10, 10, 10, 20, 20, 20, 25, 40, 40, 40, 40, 80, 80, 80, 85]
      #expected_output2 = [10, 10, 10, 20, 20, 20, 20, 40, 40, 40, 40, 80, 80, 80, 80]
      maxfiltered_signal = std.MaxFilter(width = 4)(original_signal)
      self.assertEqualVector(maxfiltered_signal, expected_output)

suite = allTests(TestMaxFilter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

