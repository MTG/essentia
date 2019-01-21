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

from essentia.standard import CoverSongSimilarity
from essentia import array, run, Pool
from essentia_test import *
import numpy as np


class TestCoverSimilarity(TestCase):

    # computed using the python implementation from https://github.com/albincorreya/ChromaCoverId/blob/master/cover_similarity_measures.py
    input_matrix = np.load('highlevel/simMatrix_mozart30sec_vivaldi.npy')
    expected_distance = 0.701402

    def testEmpty(self):
        self.assertComputeFails(CoverSongSimilarity(), [])

    def testRegression(self):
        sim = CoverSongSimilarity()
        score_matrix = sim.compute(array(self.input_matrix))
        distance = np.sqrt(self.input_matrix.shape[1]) / np.max(score_matrix)

        self.assertAlmostEqual(self.expected_distance, distance)

    def testInvalidParam(self):
        self.assertConfigureFails(CoverSongSimilarity(), { 'simType': 1 })
        self.assertConfigureFails(CoverSongSimilarity(), { 'simType': 'dmax' })

    """
    def testStreamingRegression(self):
        from essentia.streaming import CoverSongSimilarity as CoverSim

        sim = CoverSim()
        pool = Pool()
        
        array(self.input_matrix) >> sim.inputArray
        sim.inputArray >> (pool, 'score_matrix')
        run(sim)

        distance = np.sqrt(pool['score_matrix'].shape[1]) / np.max(pool['score_matrix'])

        self.assertAlmostEqual(self.expected_distance, distance)
    """

suite = allTests(TestCoverSimilarity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)