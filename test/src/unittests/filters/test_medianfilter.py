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
from math import *
from essentia import array as esarr


class TestMedianFilter(TestCase):

    def testZeros(self):
        x = esarr(numpy.zeros(1000))
        y = MedianFilter()(x)

        self.assertEqualVector(x, y)

    def testComputeFails(self):
        self.assertComputeFails(MedianFilter(kernelSize=7),
                                esarr(numpy.zeros(6)))

    def testRegression(self):
        from scipy.signal import medfilt
        kernelSize = 9
        halfKernel = kernelSize / 2

        medianFiter = MedianFilter(kernelSize=kernelSize)

        x = esarr(numpy.random.rand(1000))

        # the beginnings and ends are removed because the initialization
        # strategies are different on purpose
        yScipy = medfilt(x, kernelSize)[halfKernel:-halfKernel]
        yEssentia = medianFiter(x)[halfKernel:-halfKernel]

        self.assertAlmostEqualVector(yScipy, yEssentia, 1e-6)

    def testEvenKernelFails(self):
        self.assertConfigureFails(DiscontinuityDetector(), {'kernelSize': 12})


suite = allTests(TestMedianFilter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
