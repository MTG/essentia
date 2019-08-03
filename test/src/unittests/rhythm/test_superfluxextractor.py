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
        sampleRate = 44100
        frameSize = 2048
        hopSize = 256
        signal = zeros(sampleRate * 2)
        # impulses at 0:30 and 1:00
        signal[22050] = 1.
        signal[44100] = 1.

        expected = [0.5, 1.]

        result = SuperFluxExtractor(sampleRate=sampleRate, frameSize=frameSize,
                                    hopSize=hopSize)(signal)

        # SuperfluxPeaks has a parameter 'combine' which is a threshold that
        # puts together consecutive peaks. This means that a peak will be
        # detected as soon as it is seen by a frame. Thus, the frame size
        # also influences the expected precision of the algorithm.
        precission = (hopSize + frameSize) / sampleRate
        self.assertAlmostEqualVectorAbs(result, expected, precission)


suite = allTests(TestSuperFluxExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
