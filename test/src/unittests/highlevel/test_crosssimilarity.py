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

from essentia.standard import CrossSimilarityMatrix
from essentia import array, run, Pool
from essentia_test import *
import numpy as np


class TestCrossSimilarityMatrix(TestCase):

    query_hpcp = np.load('highlevel/mozart30sec_hpcp.npy')
    ref_hpcp = np.load('highlevel/vivaldi_hpcp.npy')
    # computed using the python implementation from https://github.com/albincorreya/ChromaCoverId/blob/master/cover_similarity_measures.py
    expected = np.load('highlevel/simMatrix_mozart30sec_vivaldi.npy')

    def testEmpty(self):
        self.assertComputeFails(CrossSimilarityMatrix(), [])

    def testRegression(self):

        csm = CrossSimilarityMatrix()
        result = csm.compute(array(self.query_hpcp), array(self.ref_hpcp))

        self.assertAlmostEqual(np.mean(self.expected), np.mean(result))
        self.assertAlmostEqualVector(self.expected, result)

    def testInvalidParam(self):
        self.assertConfigureFails(CrossSimilarityMatrix(), { 'kappa': -1 })
        self.assertConfigureFails(CrossSimilarityMatrix(), { 'otiBinary': -1 })

    def testOTIBinaryCompute(self):
        # test oti-based binary sim matirx method
        csm = CrossSimilarityMatrix(otiBinary=True)
        result = csm.compute(array(self.query_hpcp), array(self.ref_hpcp))

        self.assertComputeFails(CrossSimilarityMatrix(otiBinary=True), [])


    """
    def testStreamingRegression(self):
        from essentia.streaming import CrossSimilarityMatrix as CSM
        import essentia.streaming as es

        simMatrix = CSM()
        pool = Pool()

        vector1 = es.VectorInput(array(self.query_hpcp))
        vector2 = es.VectorInput(array(self.ref_hpcp))
        accumulator1 = es.VectorRealAccumulator()
        accumulator2 = es.VectorRealAccumulator()
        
        vector1.data >> accumulator1.data
        accumulator1.array >> simMatrix.queryFeature
        vector2.data >> accumulator2.data
        accumulator2.array >> simMatrix.referenceFeature
        simMatrix.csm >> 
        (pool, 'csm')
        run(simMatrix)

        self.assertAlmostEqual(np.mean(pool['csm']), np.mean(self.expected))
        self.assertAlmostEqualVector(pool['csm'], self.expected)
    """

suite = allTests(TestCrossSimilarityMatrix)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)