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
from random import randint


class TestEffectiveDuration(TestCase):

    def testEmpty(self):
        input = []
        self.assertEqual(EffectiveDuration()(input), 0.0)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(EffectiveDuration()(input), 0.)

    def testOne(self):
        input = [0.3]
        self.assertAlmostEqual(EffectiveDuration()(input), 1/44100.0)
        input = [0]
        self.assertAlmostEqual(EffectiveDuration()(input), 0)


        input = [100]
        self.assertAlmostEqual(EffectiveDuration()(input), 1/44100.0)

    def test30Sec(self):
        input = [randint(41, 100) for x in xrange(44100*30)]
        self.assertAlmostEqual(EffectiveDuration()(input), 30)

    def test15SecOf30Sec(self):
        input1 = [randint(41, 100) for x in xrange(44100*15)]
        input1[0] = 100 # to ensure that at least one element is 100
        input2 = [randint(0, 39) for x in xrange(44100*15)]
        input = input1 + input2

        self.assertAlmostEqual(EffectiveDuration()(input), 15)

    def testNegative20SecOf40Sec(self):
        # Note: this test assumes the thresholdRatio is 40%
        input1 = [randint(-100, -41) for x in xrange(44100*10)]
        input2 = [randint(0, 39) for x in xrange(44100*10)]
        input3 = [randint(41, 100) for x in xrange(44100*10)]
        input3[0] = 100 # to ensure that at least one element is 100
        input4 = [randint(-39, 0) for x in xrange(44100*10)]

        input = input1 + input2 + input3 + input4

        self.assertAlmostEqual(EffectiveDuration()(input), 20)

    def testBadSampleRate(self):
        self.assertConfigureFails(EffectiveDuration(), { 'sampleRate' : 0 })


suite = allTests(TestEffectiveDuration)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
