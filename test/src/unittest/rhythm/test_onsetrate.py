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
from numpy import random

sr = 44100.0
frameSize = 1024
hopSize = 512
frameRate = sr/hopSize
frameTime = 1/frameRate

class TestOnsetRate(TestCase):

    def testZero(self):
        # zeros should return no onsets
        size = 5*sr
        times, rate = OnsetRate()(zeros(size))
        self.assertEqualVector(times, [])
        self.assertEqual(rate, 0)

    def testEmpty(self):
        # Empty input returns no onsets
        self.assertComputeFails(OnsetRate(), [])

    def testConstantInput(self):
        dur = 5
        size = int(dur*sr)
        times, rate = OnsetRate()(ones(size))
        # a constant input signal should return no onsets, however as
        # the last frame contains less samples (although it is zeropadded)
        # therefore in a constant input signal it will always detect an onset
        # on the last frame. Maybe last frame should be skipped in the
        # algorithm?
        # TODO why not having an onset in the first frame? it seems logical 
        # to have it there for non-zero constant signals
        firstFrame = 1*frameTime
        lastFrame = (int(size/hopSize)-1)*frameTime
        self.assertAlmostEqualVector(times, [firstFrame, lastFrame])
        self.assertAlmostEqual(rate, 2.0/dur)

    def makeImpulse(self, list, pos):
        if pos >= len(list): return
        if pos-1 >= 0: list[pos-1] = 0.2
        if pos+1 < len(list): list[pos] = 0.3
        list[pos] = 1.0

    def testImpulse(self):
        # Given an impulse should return its position
        dur = 2
        size = int(dur*sr)
        signal = ones(size)*0.1
        for i in range(size):
            signal[i] += random.rand(1)*0.01
        expectedTimes = [0.01, 0.17, 0.73, 0.99]
        for i in range(len(expectedTimes)):
            expectedTimes[i] *= dur
            self.makeImpulse(signal, int(expectedTimes[i]*44100.0))
        expectedRate = float(len(expectedTimes))/float(dur)
        times, rate = OnsetRate()(signal)
        self.assertAlmostEqual(rate, expectedRate)
        # to compare seconds it is better to compare the difference, such as
        # if is less than 50ms
        diff = abs(times - expectedTimes)
        self.assertAlmostEqualVector(diff, zeros(len(times)), 0.05)




suite = allTests(TestOnsetRate)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
