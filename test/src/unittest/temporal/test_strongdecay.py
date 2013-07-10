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
from math import sqrt
from math import fabs

class TestStrongDecay(TestCase):

    def testEmpty(self):
        self.assertComputeFails(StrongDecay(), [])

    def testOne(self):
        self.assertComputeFails(StrongDecay(), [1])

    def testFlat(self):
        signal = [.5]*10
        centroid = sum(range(len(signal))) / float(len(signal))
        relativeCentroid = centroid * (1 / 44100.)
        energy = sum([x**2 for x in signal])
        strongDecay = sqrt(energy/relativeCentroid)
        self.assertAlmostEqual(
                StrongDecay()(signal),
                strongDecay)

    def testZero(self):
        self.assertComputeFails(StrongDecay(), [0]*10)

    def testSanity(self):
        growingSignal = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
        decayingSignal = [1,.9,.8,.7,.6,.5,.4,.3,.2,.1]

        self.assertTrue(StrongDecay()(growingSignal) <
                        StrongDecay()(decayingSignal))

    def testRegression(self):
        self.assertAlmostEqual(StrongDecay()([1,.9,.8,.7,.6,.5,.4,.3,.2,.1]), 237.897033691)

    def testZeroSumSignal(self):
        signal = [1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
        centroid = sum([i*fabs(x) for i,x in enumerate(signal)])/sum([fabs(x) for x in signal])
        energy = sum([x*x for x in signal])
        self.assertAlmostEqual(StrongDecay(sampleRate=1)(signal),
                               sqrt(float(energy)/float(centroid)))

suite = allTests(TestStrongDecay)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
