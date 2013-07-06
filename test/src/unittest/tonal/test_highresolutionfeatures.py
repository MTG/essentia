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

class TestHighResolutionFeatures(TestCase):

    def testEmpty(self):
        # empty array should throw an exception cause it is not multiple of 12:
        self.assertComputeFails(HighResolutionFeatures(), [])

    def testZero(self):
        # array of zeros should output zeros
        self.assertEqualVector(HighResolutionFeatures()(zeros(120)), [0, 0, 0])

    def testPerfectlyTempered(self):
        nCents = 10
        nSemitones = 12
        size = nCents*nSemitones
        hpcp = zeros(size)
        # peaks at semitone position:
        for i in range(0, size, nCents):
            hpcp[i] = 1.0
        self.assertEqualVector(HighResolutionFeatures()(hpcp), [0, 0, 0])

    def testMaxDeviation(self):
        nCents = 10
        nSemitones = 12
        size = nCents*nSemitones
        hpcp = zeros(size)
        # peaks at semitone position + 0.5:
        for i in range(0, size, nCents/2):
            hpcp[i] = 1.0
        self.assertEqualVector(HighResolutionFeatures()(hpcp), [0.25, 0.5, 0.5])

    def testStreamingMaxDeviation(self):
        from essentia.streaming import HighResolutionFeatures as\
        strHighResolutionFeatures

        nCents = 10
        nSemitones = 12
        size = nCents*nSemitones
        hpcp = [0]*size
        # peaks at semitone position + 0.5:
        for i in range(0, size, nCents/2):
            hpcp[i] = 1

        gen = VectorInput([hpcp])
        hrf = strHighResolutionFeatures()
        pool = Pool()
        gen.data >> hrf.hpcp
        hrf.equalTemperedDeviation >> (pool, "deviation")
        hrf.nonTemperedEnergyRatio >> (pool, "energyRatio")
        hrf.nonTemperedPeaksEnergyRatio >> (pool, "peaksRatio")
        run(gen)

        self.assertEqual(pool['deviation'], 0.25)
        self.assertEqual(pool['energyRatio'], 0.5)
        self.assertEqual(pool['peaksRatio'], 0.5)


suite = allTests(TestHighResolutionFeatures)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
