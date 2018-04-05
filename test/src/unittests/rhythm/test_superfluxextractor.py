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


class TestSuperFluxExtractor(TestCase):

    def testSilence(self):
        # zeros should return no onsets (empty array)
        self.assertEqualMatrix(SuperFluxExtractor()(zeros(44100)), [])

    def testEmpty(self):
        # empty input should return no onsets (empty array)
        self.assertEqualMatrix(SuperFluxExtractor()([]), [])
        # Empty input should raise an exception
        self.assertComputeFails(Onsets(), array([[]]), [])

    def testImpulse(self):
        # Given an impulse should return its position
        signal = zeros(44100*2)
        # impulses at 0:30 and 1:00
        signal[22050] = 1.
        signal[44100] = 1.

        expected = [0.5, 1.]
        result = SuperFluxExtractor(frameSize=2048, hopSize=256)(signal)

        # TODO: what is the allowed precision? I would expect it to be 256./44100
        #       in an ideal case, but what about practice?
        self.assertAlmostEqualVectorAbs(result, expected, 256./44100)


suite = allTests(TestSuperFluxExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
