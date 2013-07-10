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

class TestZeroCrossingRate(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(ZeroCrossingRate(), input)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testOne(self):
        input = [0]
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

        input = [100]
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testAllPositive(self):
        input = [1]*100
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testAllNegative(self):
        input = [-1]*100
        self.assertAlmostEqual(ZeroCrossingRate()(input), 0)

    def testRegression(self):
        input = [45, 78, 1, -5, -.1125, 1.236, 10.25, 100, 9, -78]
        self.assertAlmostEqual(ZeroCrossingRate()(input), 3./10.)

    def testRealCase(self):
        # a 5000 cycle sine wave should cross the zero line 10000 times
        filename=join(testdata.audio_dir, 'generated', 'sine_440_5000period.wav')
        signal = MonoLoader(filename=filename)()
        wavzcr = ZeroCrossingRate(threshold=0.0)(signal)*len(signal)

        filename=join(testdata.audio_dir, 'generated', 'sine_440_5000period.mp3')
        signal = MonoLoader(filename=filename)()
        mp3zcr = ZeroCrossingRate(threshold=0.01)(signal)*len(signal)

        self.assertAlmostEqual(wavzcr, 10000)
        self.assertAlmostEqual(wavzcr, mp3zcr)


suite = allTests(TestZeroCrossingRate)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
