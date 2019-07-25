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

class TestHistogram(TestCase):

  def testZero(self):
    histogram, binEdges = Histogram(normalize="none", maxValue=1., minValue=0., numberBins=10)(zeros(1000))
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
    self.assertConfigureFails(Histogram(), {'minValue' : 1, 'maxValue' : 1, 'numberBins' : 2})

  def testRegression(self):
    inputArray = readVector(join(filedir(), 'stats/input.txt'))
    expectedEdges = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]
    expectedHistogramNone = [113,  87,  98, 104, 114,  86,  99,  88, 102, 109]
    expectedHistogramUnitSum = [ 0.113,  0.087,  0.098,  0.104,  0.114,  0.086,  0.099,  0.088,  0.102,  0.109]  
    expectedHistogramUnitMax = [ 0.99122807,  0.76315789,  0.85964912,  0.9122807 ,  1., 0.75438596,  0.86842105,  0.77192982,  0.89473684,  0.95614035] 

    (outputHistogramNone, outputEdgesNone) = Histogram(normalize="none", numberBins=10, minValue=0., maxValue=1.)(inputArray)
    (outputHistogramUnitSum, outputEdgesUnitSum) = Histogram(normalize="unit_sum", numberBins=10, minValue=0., maxValue=1.)(inputArray)
    (outputHistogramUnitMax, outputEdgesUnitMax) = Histogram(normalize="unit_max", numberBins=10, minValue=0., maxValue=1.)(inputArray)

    self.assertAlmostEqualVector(outputEdgesNone, expectedEdges, 0.001)
    self.assertAlmostEqualVector(outputHistogramNone, expectedHistogramNone, 0.001)
    self.assertAlmostEqualVector(outputEdgesUnitSum, expectedEdges, 0.001)
    self.assertAlmostEqualVector(outputHistogramUnitSum, expectedHistogramUnitSum, 0.001)
    self.assertAlmostEqualVector(outputEdgesUnitMax, expectedEdges, 0.001)
    self.assertAlmostEqualVector(outputHistogramUnitMax, expectedHistogramUnitMax, 0.001)

suite = allTests(TestHistogram)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
