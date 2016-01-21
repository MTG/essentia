#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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


class TestOnsets(TestCase):

    def testZero(self):
        # zeros should return no onsets (empty array)
        n = 10
        detection = zeros(100).reshape(n,n)
        weights = ones(n)
        self.assertEqualMatrix(Onsets()(detection, weights), [])

    def testConstantInput(self):
        # constant detection function should return first position:
        n = 10
        detection = ones(100).reshape(n,n)
        weights = ones(n)
        size = 2048
        sr = 44100.0
        while (size > 32):
            hopsize = size/2
            # TODO there will be an onset detected on the first frame for a 
            # non-zero constant signal, which is probably ok
            frameTime = (size - hopsize)/sr
            frameRate = 1.0/frameTime
            if (size == 1024 and hopsize == 512):
                onset = Onsets(frameRate=frameRate)(detection, weights)
                self.assertAlmostEqualVector(onset, array([frameTime]))
            else:
                # Onsets does not support other framerates than
                # (1024-512)/44100
                # Onsets() outputs a warning instead of exception from now on
                # self.assertConfigureFails(Onsets(), { 'frameRate': frameRate })
                pass
            size /= 2

    def testImpulse(self):
        # Given an impulse should return its position
        n = 10
        detection = zeros(100).reshape(n,n)
        for i in range(len(detection)):
            detection[i, 5] = 1
            detection[i, 4] = .2
            detection[i, 6] = .3
        weights = ones(n)
        size = 2048
        sr = 44100.0
        while (size > 32):
            hopsize = size/2
            frameTime = (size - hopsize)/sr
            frameRate = 1.0/frameTime
            if (size == 1024 and hopsize == 512):
                onset = Onsets(frameRate=frameRate)(detection, weights)
                self.assertAlmostEqualVector( onset, array([4*frameTime]), 1e-5)
            else:
                # Onsets does not support other framerates than
                # (1024-512)/44100
                # self.assertConfigureFails(Onsets(), { 'frameRate': frameRate })
                pass # from now on Onset returns a warning instead of exception
            size /= 2

    def testInvalidParam(self):
        self.assertConfigureFails(Onsets(), { 'frameRate':-1 })
        self.assertConfigureFails(Onsets(), { 'alpha': 2 })
        self.assertConfigureFails(Onsets(), { 'delay': -1 })
        self.assertConfigureFails(Onsets(), { 'silenceThreshold':10 })

    def testEmpty(self):
        # Empty input should raise an exception
        self.assertComputeFails(Onsets(), array([[]]), [])


    def testBadWeightSize(self):
        weights = [1,2,3,4]
        input = [[1,2,3,4],
                 [5,6,7,8],
                 [9,10,11,12]]

        self.assertComputeFails(Onsets(), input, weights)


suite = allTests(TestOnsets)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
