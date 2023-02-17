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
from essentia import run as ess_run
import essentia.streaming as ess
import numpy as np


class TestChromaCrossSimilarity(TestCase):
    
    # hpcp matrix of a short query song segment (2 frames) computed using essentia hpcp algorithm
    query_hpcp = array([[0.3218126, 0.00541916, 0.26444072, 0.36874822, 1., 0.10472599, 0.05123469, 0.03934194, 0.07354275, 0.646091, 0.55201685, 0.03270169],
                    [0.07695414, 0.04679213, 0.56867135, 1., 0.10247268, 0.03653419, 0.03635696, 0.2443251, 0.2396715, 0.1190474, 0.8045795, 0.41822678]])
    
    # hpcp matrix of a short reference song segment (3 frames) computed using essentia hpcp algorithm
    reference_hpcp = array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0.36084786, 0.37151814, 0.40913638, 0.15566002, 0.40571737, 1., 0.6263613, 0.65415925, 0.53127843, 0.7900088, 0.50427467, 0.51956046],
                    [0.42861825, 0.36887613, 0.05665652, 0.20978431, 0.1992704, 0.14884946, 1., 0.24148795, 0.43031794, 0.14265466, 0.17224492, 0.36498153]]) 
    
    # expected binary similarity matrix using the cross recurrence quantification method with oti computed using the python implementation from https://github.com/albincorreya/ChromaCoverId
    expected_crp_simmatrix = array([[0., 0., 1.],
                                    [0., 0., 0.]])

    # expected binary similarity matrix using oti-based similarity method using the python implementation from https://github.com/albincorreya/ChromaCoverId
    expected_oti_simmatrix = array([[1., 0., 0.], 
                                    [1., 0., 0.]])

    def testEmpty(self):
        self.assertComputeFails(ChromaCrossSimilarity(otiBinary=False, frameStackSize=1), [], [])
        self.assertComputeFails(ChromaCrossSimilarity(otiBinary=True, frameStackSize=1), [], [])

    def testRegressionStandard(self):
        """Test standard ChromaCrossSimilarity algo rqa method with 'oti=True'"""
        csm = ChromaCrossSimilarity(frameStackSize=1)
        result_simmatrix = csm(self.query_hpcp, self.reference_hpcp)
        self.assertAlmostEqualMatrix(self.expected_crp_simmatrix, result_simmatrix)

    def testRegressionOTIBinary(self):
        """Test regression of standard ChromaCrossSimilarity when otiBinary=True"""
        csm = ChromaCrossSimilarity(otiBinary=True, frameStackSize=1)
        sim_matrix = csm(self.query_hpcp, self.reference_hpcp)
        self.assertAlmostEqual(np.mean(self.expected_oti_simmatrix), np.mean(sim_matrix))
        self.assertAlmostEqualMatrix(self.expected_oti_simmatrix, sim_matrix)

    def testRegressionStreaming(self):
        """Tests streaming ChromaCrossSimilarity algo with 'otiBinary=True' against the standard mode algorithm with 'otiBinary=True' """
        # compute chromacrosssimilarity matrix using streaming mode
        queryVec = ess.VectorInput(self.query_hpcp)
        csm_streaming = ess.ChromaCrossSimilarity(referenceFeature=self.reference_hpcp, oti=0, frameStackSize=1)
        pool = Pool()
        queryVec.data >> csm_streaming.queryFeature
        csm_streaming.csm >>  (pool, 'csm')

        ess_run(queryVec)

        self.assertAlmostEqualMatrix(self.expected_oti_simmatrix, np.array(pool['csm']))
    

suite = allTests(TestChromaCrossSimilarity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

