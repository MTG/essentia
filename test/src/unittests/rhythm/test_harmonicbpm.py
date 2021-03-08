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
import essentia
from essentia import Pool
from essentia.standard import *
import essentia.streaming as ess

from math import ceil, fabs
from numpy import argmax


class TestHarmonicBpm(TestCase):

    def testInvalidParam(self):
        # Test that we must give valid frequency ranges or order
        self.assertConfigureFails(HarmonicBpm(), {'bpm': 0})
        self.assertConfigureFails(HarmonicBpm(), { 'threshold': 0 })
        self.assertConfigureFails(HarmonicBpm(), { 'tolerance': -1 })

    def testConstantInput(self):
        # constant input reports bpms 0
        testBpms = [120,120,120,120,120,120,120]
        harmonicBpms = HarmonicBpm(bpm=120)(testBpms)
        self.assertEqual(harmonicBpms, 120)

    """
    FIXME-This TC currently cause the program to hang.
    def testZeros(self):
        # constant input reports bpms 0
        testBpms = [0,0,0,0,0,0,0,0,0,0,0]
        self.assertRaises(RuntimeError, lambda: HarmonicBpm()(testBpms))
    """

    def testRegressionVariableTempoImpulse(self):
        testBpms = [100,101,102,103,104]
        harmonicBpms = HarmonicBpm(bpm=100)(testBpms)
        expectedBpm=102
        self.assertAlmostEqual(harmonicBpms, expectedBpm, 1)

    def testRegressionVariableTempoImpulse(self):
        testBpms = [100,101,102,103,104,200,202,204,206,208]
        harmonicBpms = HarmonicBpm(bpm=100)(testBpms)
        expectedBpm=[100.0,200.0]
        self.assertEqualVector(harmonicBpms, expectedBpm)

    def testEmpty(self):
        # nothing should be computed and the resulting pool be empty
        harmonicBpms = HarmonicBpm(bpm=100)([])
        self.assertEqualVector(harmonicBpms, [])

suite = allTests(TestHarmonicBpm)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
