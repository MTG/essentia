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

from essentia_test import *


class TestCoverSongSimilarity(TestCase):
    '''Unit tests for essentia CoverSongSimilarity algorithm'''    
    # pre-defined binary similarity matrix for the test
    sim_matrix = array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]])
    # expected cover similarity distance
    expected_distance = 1.732

    def testEmpty(self):
        self.assertComputeFails(CoverSongSimilarity(), [])

    def testRegressionStandard(self):
        '''Test regression of CoverSongSimilarity algorithm in standard mode'''
        sim = CoverSongSimilarity()
        score_matrix, distance = sim.compute(self.sim_matrix)
        self.assertAlmostEqualFixedPrecision(self.expected_distance, distance)
        warn = "Expected shape of output score_matrix is %s, instead of %s" % (self.sim_matrix.shape, score_matrix.shape)
        self.assertEqual(score_matrix.shape[0], self.sim_matrix.shape[0], warn)
        self.assertEqual(score_matrix.shape[1], self.sim_matrix.shape[1], warn)

    def testInvalidParam(self):
        self.assertConfigureFails(CoverSongSimilarity(), { 'distanceType': 'test' })
        self.assertConfigureFails(CoverSongSimilarity(), { 'alignmentType': 'test' })

    def testRegressionStreaming(self):
        '''Test regression of CoverSongSimilarity algorithm in streaming mode'''
        from essentia.streaming import CoverSongSimilarity as CoverSongSimilarityStreaming

        matrix_input = VectorInput(self.sim_matrix)
        coversim_streaming = CoverSongSimilarityStreaming(pipeDistance=True)
        pool = Pool()
        matrix_input.data >> coversim_streaming.inputArray
        coversim_streaming.scoreMatrix >>  (pool, 'scoreMatrix')
        coversim_streaming.distance >> (pool, 'distance')
        # run the algorithm network
        run(matrix_input)

        self.assertAlmostEqualFixedPrecision(self.expected_distance, pool['distance'][-1])


suite = allTests(TestCoverSongSimilarity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

