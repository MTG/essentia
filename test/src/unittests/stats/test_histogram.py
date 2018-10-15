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

testdir = join(filedir(), 'histogram')

class TestHistogram(TestCase):

  def testZero(self):
    self.assertEqualVector(Histogram()(zeros(1000)), [[1000], [0.5]])

  def testOutOfRangeConfiguration(self):
    self.assertConfigureFails(Histogram(), {'normalize' : 'y'})
    self.assertConfigureFails(Histogram(), {'maxValue' : -1})
    self.assertConfigureFails(Histogram(), {'minValue' : -1})
    self.assertConfigureFails(Histogram(), {'numberBins' : 0})
    self.assertConfigureFails(Histogram(), {'numberBins' : -1})

  def testInvalidConfigurationCombination(self):
    self.assertRaises(EssentiaException, Histogram(maxValue = 0, minValue = 1), (zeros(1000)))

  def testRegression(self):
   inputArray = readVector(join(filedir(), 'stats/input.txt'))
   expectedRangesNone = readVector(join(testdir, 'ranges.txt'))
   expectedCountsNone = readVector(join(testdir, 'counts.txt'))

   (outputCountsNone, outputRangesNone) = Histogram(numberBins = 10, minValue = 0.000039890, maxValue = 0.99970)(inputArray)

   self.assertAlmostEqualVector(outputRangesNone, expectedRangesNone, 0.01)
   self.assertAlmostEqualVector(outputCountsNone, expectedCountsNone, 0.01)

suite = allTests(TestHistogram)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
