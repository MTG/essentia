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

from essentia import array
from essentia_test import *


class TestCrossSimilarityMatrix(TestCase):

    # hpcp matrix of a short query song segment (2 frames) computed using essentia hpcp algorithm
    query_feature = array([[0.3218126, 0.00541916, 0.26444072, 0.36874822, 1., 0.10472599, 0.05123469, 0.03934194, 0.07354275, 0.646091, 0.55201685, 0.03270169],
                    [0.07695414, 0.04679213, 0.56867135, 1., 0.10247268, 0.03653419, 0.03635696, 0.2443251, 0.2396715, 0.1190474, 0.8045795, 0.41822678]])
    
    # hpcp matrix of a short reference song segment (3 frames) computed using essentia hpcp algorithm
    reference_feature = array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0.36084786, 0.37151814, 0.40913638, 0.15566002, 0.40571737, 1., 0.6263613, 0.65415925, 0.53127843, 0.7900088, 0.50427467, 0.51956046],
                    [0.42861825, 0.36887613, 0.05665652, 0.20978431, 0.1992704, 0.14884946, 1., 0.24148795, 0.43031794, 0.14265466, 0.17224492, 0.36498153]]) 
    
    # expected euclidean pairwise similarity matrix without binary thresholding (pre-computed using a python script adopted from https://github.com/albincorreya/ChromaCoverId/blob/master/cover_similarity_measures.py)
    expected_sim_matrix = [[1.432924 , 1.5921365, 1.5593135],
                           [1.5159905, 1.7596511, 1.5824637]]
    # expected euclidean pairwise similarity matrix with binary thresholding where binarizePercentile=0.095, frameStackStride=1 and frameStackSize=1 (pre-computed using a python script adopted from https://github.com/albincorreya/ChromaCoverId/blob/master/cover_similarity_measures.py)
    expected_sim_matrix_binary = [[1., 0., 0.],
                                  [0., 0., 0.]]

    def testEmpty(self):
        self.assertComputeFails(CrossSimilarityMatrix(), [], [])

    def testRegressionStandard(self):
        csm = CrossSimilarityMatrix(binarize=False, frameStackStride=1, frameStackSize=1)
        result = csm(self.query_feature, self.reference_feature)
        self.assertAlmostEqualMatrix(self.expected_sim_matrix, result)

    def testRegressionBinary(self):
        csm = CrossSimilarityMatrix(binarize=True, binarizePercentile=0.095, frameStackStride=1, frameStackSize=1)
        result = csm(self.query_feature, self.reference_feature)
        self.assertAlmostEqualMatrix(self.expected_sim_matrix_binary, result)


suite = allTests(TestCrossSimilarityMatrix)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
