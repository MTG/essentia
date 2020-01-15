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

from sklearn.metrics.pairwise import euclidean_distances
from essentia.standard import CrossSimilarityMatrix
from essentia import array
from essentia_test import *
import numpy as np


def cross_similarity_matrix(input_x, input_y, binarize=True, tau=1, m=1, kappa=0.095):
    pdistances = euclidean_distances(input_x, input_y)
    if binarize:
        transposed_pdistances = pdistances.T
        eph_x = np.percentile(pdistances, kappa*100, axis=1)
        eph_y = np.percentile(transposed_pdistances, kappa*100, axis=1)
        x = eph_x[:,None] - pdistances
        y = eph_y[:,None] - transposed_pdistances
        #apply heaviside function to the array (Binarize the array)
        x = np.piecewise(x, [x<0, x>=0], [0,1])
        y = np.piecewise(y, [y<0, y>=0], [0,1])
        crp = x*y.T
        return crp
    else:
        return pdistances


class TestChromaCrossSimilarity(TestCase):

    #query = estd.MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'mozart_c_major_30sec.wav'))()
    #reference = estd.MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'Vivaldi_Sonata_5_II_Allegro.wav'))
    query_feature = array(np.ones((100,200)))
    reference_feature = array(np.zeros((200,100)))
    expected = cross_similarity_matrix(query_feature, reference_feature, binarize=False)
    expected_binary = cross_similarity_matrix(query_feature, reference_feature, binarize=True)

    def testEmpty(self):
        self.assertComputeFails(CrossSimilarityMatrix(), [])

    def testRegressionStandard(self):
        csm = CrossSimilarityMatrix(binarize=False)
        result = csm.compute(self.query_feature, self.reference_feature)
        self.assertAlmostEqual(np.mean(self.expected), np.mean(result))
        self.assertAlmostEqualVector(self.expected, result)

    def testRegressionBinary(self):
        csm = CrossSimilarityMatrix(binarize=True, binarizePercentile=0.095)
        result = csm.compute(self.query_feature, self.reference_feature)
        self.assertAlmostEqualVector(self.expected_binary, result)

    def testInvalidParam(self):
        self.assertConfigureFails(CrossSimilarityMatrix(), { 'binarizePercentile': -1 })
        self.assertConfigureFails(CrossSimilarityMatrix(), { 'frameStackSize': True })


suite = allTests(TestCrossSimilarityMatrix)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
