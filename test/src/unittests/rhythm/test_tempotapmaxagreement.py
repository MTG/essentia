#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
import math
from essentia.standard import *
import essentia.streaming as ess
import numpy as np


class TestTempoTapMaxAgreement(TestCase):

    def testRegression(self):
        tickCandidates = [[1.0, 3.0, 5.0, 7.0, 9.0], [1.1, 3.1 , 5.1 , 7.1 , 9.3], [1.0, 3.0, 5.0, 7.0, 9.0], [0.9, 2.9, 4.9, 6.9, 8.9]]
        ticks, confidence = TempoTapMaxAgreement()(np.array(tickCandidates))
        expectedTicks = [1.0, 3.0, 5.0, 7.0, 9.0]    
        expectedConfidence = 4.5 #  Trials have shown 4.5 to be a typical ballpark value
        self.assertEqualVector(ticks, expectedTicks)                 
        self.assertAlmostEqual(confidence, expectedConfidence, 0.1)    
    
    def testZero(self):
        tickCandidates = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]] 
        # A Runtime error occurs when all values are all zero
        self.assertRaises(RuntimeError, lambda: TempoTapMaxAgreement()(np.array(tickCandidates)))
     
    def testDuplicateTickValues(self):
        tickCandidates = [[5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0, 5.0]]
        # A Runtime error occurs when all tick values in an array are the same
        self.assertRaises(RuntimeError, lambda: TempoTapMaxAgreement()(np.array(tickCandidates)))     

    def testEmpty(self):
        tickCandidates = [[], [], [], [], []] 
        ticks, confidence = TempoTapMaxAgreement()(np.array(tickCandidates))
        self.assertEqualVector(ticks, [] )                 
        self.assertEqual(confidence, 0)    

    def testNull(self):
        tickCandidates = [] 
        # A Runtime error occurs in a ttoally empty situation
        self.assertRaises(RuntimeError, lambda: TempoTapMaxAgreement()(np.array(tickCandidates)))

    def testDuplicates(self):
        tickCandidates = [[5.0, 6.0, 7.0, 8.0, 9.0], [5.0, 6.0, 7.0, 8.0, 9.0], [5.0, 6.0, 7.0, 8.0, 9.0], [5.0, 6.0, 7.0, 8.0, 9.0], [5.0, 6.0, 7.0, 8.0, 9.0]]
        ticks,confidence = TempoTapMaxAgreement()(np.array(tickCandidates))
        expectedTicks = [5.0, 6.0, 7.0, 8.0, 9.0]
        expectedConfidence = 5.32 # Max. value for confidence (see documentation)
        self.assertEqualVector(ticks, expectedTicks)                 
        self.assertAlmostEqual(confidence, expectedConfidence, 0.1)

    def testIllegalDecreasing(self):
        tickCandidates = [[9.0, 7.0, 5.0, 3.0, 1.0], [9.1, 7.1, 5.1, 3.1, 1.1], [9.2, 7.3, 5.1, 3.1, 1.1]]
        self.assertRaises(RuntimeError, lambda: TempoTapMaxAgreement()(np.array(tickCandidates)))


suite = allTests(TestTempoTapMaxAgreement)

if __name__ == '__main__':
    TextTestRunner(verbosity = 2).run(suite)
