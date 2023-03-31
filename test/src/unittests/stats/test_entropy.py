#!/usr/bin/env python

# Copyright (C) 2006-2022  Music Technology Group - Universitat Pompeu Fabra
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
import os.path
import random

from essentia_test import *
import numpy as np
import math


class TestEntropy(TestCase):

    def testEmpty(self):
        with self.assertRaises(EssentiaException):
            Entropy()(np.empty(0, dtype=np.single))

    def testNegative(self):
        with self.assertRaises(EssentiaException):
            Entropy()(np.array([-1], dtype=np.single))

        with self.assertRaises(EssentiaException):
            Entropy()(np.sin(np.linspace(0, np.pi * 1000, num=44100, dtype=np.single)))

    def testConstructedData(self):
        # Array with identical values
        self.assertAlmostEqual(0.0, Entropy()(np.zeros(100, dtype=np.single)))
        self.assertAlmostEqual(-math.log2(0.01), Entropy()(np.ones(100, dtype=np.single)), precision=1e-5)
        self.assertAlmostEqual(-math.log2(0.01), Entropy()(np.full(100, 5, dtype=np.single)), precision=1e-5)

        # Trivial distribution
        arr = np.zeros(100, dtype=np.single)
        arr[0] = 100
        self.assertAlmostEqual(0.0, Entropy()(arr))
        del arr

    def testRandomData(self):
        def calc_entropy(arr):
            arr /= np.sum(arr)
            return -np.sum(np.nan_to_num(np.log2(arr)) * arr)

        for _ in range(10):
            arr = np.array([random.random() for i in range(100)], dtype=np.single)
            self.assertAlmostEqual(calc_entropy(arr), Entropy()(arr), precision=1e-5)


suite = allTests(TestEntropy)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
