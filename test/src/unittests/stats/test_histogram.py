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
    histogram, binCenters = Histogram()(zeros(1000))
    self.assertEqualVector(histogram, [1000., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    self.assertAlmostEqualVector(binCenters, [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], 0.0001) 

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
    expectedCentersNone = readVector(join(testdir, 'ranges.txt'))
    expectedHistogramNone = readVector(join(testdir, 'counts.txt'))

    (outputHistogramNone, outputCentersNone) = Histogram(numberBins = 10, minValue = 0.000039890, maxValue = 0.99970)(inputArray)

    self.assertAlmostEqualVector(outputCentersNone, expectedCentersNone, 0.01)
    self.assertAlmostEqualVector(outputHistogramNone, expectedHistogramNone, 0.01)

suite = allTests(TestHistogram)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
