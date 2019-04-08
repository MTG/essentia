#!/usr/bin/env python

# Copyright (C) 2006-2017  Music Technology Group - Universitat Pompeu Fabra
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

from essentia.streaming import CoverSongSimilarity, VectorInput
from essentia.standard import CoverSongSimilarity as CoverSim
from essentia import array, run, Pool
from essentia_test import *
import numpy as np


class TestCoverSongSimilarity(TestCase):

    sim_matrix = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]])
    expected_distance = 1.732

    def testEmpty(self):
        self.assertComputeFails(CoverSim(), [])

    def testRegressionStandard(self):
        sim = CoverSim()
        score_matrix, distance = sim.compute(array(self.sim_matrix))
        self.assertAlmostEqual(self.expected_distance, distance)
        warn = "Expected shape of output score_matrix is %s, instead of %s" % (self.sim_matrix.shape, score_matrix.shape)
        self.assertEqual(score_matrix.shape[0], self.sim_matrix.shape[0], warn)
        self.assertEqual(score_matrix.shape[1], self.sim_matrix.shape[1], warn)

    def testInvalidParam(self):
        self.assertConfigureFails(CoverSim(), { 'alignmentType': 1 })
        self.assertConfigureFails(CoverSim(), { 'alignmentType': 'dmax' })

    def testRegressionStreaming(self):
        sim = CoverSongSimilarity()
        pool = Pool()
        inputVec = VectorInput(array(self.sim_matrix))
        inputVec.data >> sim.inputArray
        sim.scoreMatrix >> (pool, 'score_matrix')
        sim.distance >> (pool, 'distance')
        run(sim)
        distance = np.sqrt(self.sim_matrix.shape[1]) / np.max(pool['score_matrix'])
        self.assertAlmostEqual(self.expected_distance, distance)


suite = allTests(TestCoverSongSimilarity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

