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
from math import log10

class TestLogAttackTime(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(LogAttackTime(), input)

    def testSilence(self):
        input = [0]*100
        self.assertAlmostEqual(LogAttackTime()(input), -5)

    def testOne(self):
        input = [0]
        self.assertEqual(LogAttackTime()(input), -5)

        input = [100]
        self.assertEqual(LogAttackTime()(input), -5)

    def testInvalidStartStop(self):
        self.assertConfigureFails(
                LogAttackTime(),
                {'startAttackThreshold': .8, 'stopAttackThreshold': .2})

    def testImpulse(self):
        input = [1,1,1,1,1,10,1,1,1,1,1]

        self.assertEqual( LogAttackTime()(input), -5 )

    def testRegression(self):
        #       start                              stop
        #         |---------------------------------|
        input = [45, 78, 1, 5, .1125, 1.236, 10.25, 100, 9, 78]
        expected = log10(7/44100.)

        self.assertAlmostEqual(
                LogAttackTime(startAttackThreshold=.1,
                              stopAttackThreshold=.9)(input),
                expected)


    def testZero(self):
        input = [0]*1024
        self.assertEqual(LogAttackTime()(input), -5)


suite = allTests(TestLogAttackTime)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
