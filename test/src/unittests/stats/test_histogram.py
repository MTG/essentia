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
    histogram, binEdges = Histogram(normalize = "none", maxValue = 1., minValue = 0., numberBins = 10)(zeros(1000))
    self.assertEqualVector(histogram, [1000., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    self.assertAlmostEqualVector(binEdges, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.0001) 

  def testOutOfRangeConfiguration(self):
    self.assertConfigureFails(Histogram(), {'normalize' : 'y'})
    self.assertConfigureFails(Histogram(), {'maxValue' : -1})
    self.assertConfigureFails(Histogram(), {'minValue' : -1})
    self.assertConfigureFails(Histogram(), {'numberBins' : 0})
    self.assertConfigureFails(Histogram(), {'numberBins' : -1})

  def testInvalidConfigurationCombination(self):
    self.assertConfigureFails(Histogram(), {'minValue' : 1, 'maxValue' : 0})

  def testRegression(self):
    inputArray = readVector(join(filedir(), 'stats/input.txt'))
    expectedEdgesNone = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]
    expectedHistogramNone = [113,  87,  98, 104, 114,  86,  99,  88, 102, 109]

    (outputHistogramNone, outputEdgesNone) = Histogram(numberBins = 10, minValue = 0., maxValue = 1.)(inputArray)

    self.assertAlmostEqualVector(outputEdgesNone, expectedEdgesNone, 0.001)
    self.assertAlmostEqualVector(outputHistogramNone, expectedHistogramNone, 0.001)

suite = allTests(TestHistogram)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
