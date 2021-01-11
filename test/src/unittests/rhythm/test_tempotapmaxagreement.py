#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

    def testRegression(self, tempotapmaxagreement = None):
        # to match with test_tempotap.py
        tickCandidates = [[5.0,6.0,7.0,8.0,9.0],[15.0,16.0,17.0,18.0,19.0],[25.0,35.0,45.0,55.0,65.0],[45.0,46.0,47.0,48.0,49],[81.0,82.0,85.0,88.0,92.0]] 
        ticks,confidence = TempoTapMaxAgreement()(np.array(tickCandidates))
        expectedTicks= [81., 82., 85., 88., 92.]
        expectedConfidence=4.11
        self.assertEqualVector(ticks, expectedTicks)                 
        self.assertAlmostEqual(confidence, expectedConfidence,0.1)    
    
    def testZero(self):
        tickCandidates = [[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]] 
        # The values in teh above vectors shgould be increasing. If they are all the same a Runtime error occurs
        self.assertRaises(RuntimeError, lambda: TempoTapMaxAgreement()(np.array(tickCandidates)))
     
    def testEmpty(self):
        tickCandidates = [[],[],[],[],[]] 
        ticks,confidence = TempoTapMaxAgreement()(np.array(tickCandidates))
        self.assert_(all(array(ticks) == 0.0))
        self.assertEqual(confidence, 0.0)
    
    def testDuplicates(self):
        tickCandidates = [[5.0,6.0,7.0,8.0,9.0],[5.0,6.0,7.0,8.0,9.0],[5.0,6.0,7.0,8.0,9.0],[5.0,6.0,7.0,8.0,9.0],[5.0,6.0,7.0,8.0,9.0]] 
        ticks,confidence = TempoTapMaxAgreement()(np.array(tickCandidates))
        expectedTicks= [5.0,6.0,7.0,8.0,9.0]
        expectedConfidence=5.0
        self.assertEqualVector(ticks, expectedTicks)                 
        self.assertAlmostEqual(confidence, expectedConfidence,0.1)

suite = allTests(TestTempoTapMaxAgreement)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
    
