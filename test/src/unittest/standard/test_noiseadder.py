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

class TestNoiseAdder(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(NoiseAdder(), { 'level': 10 })

    def testLevel(self):
        ng = NoiseAdder(level = 0)
        level_0 = ng(zeros(1000))
        self.assert_(all(abs(level_0) <= 1))

        ng.configure(level = -10)
        level_0_1 = ng(zeros(1000))
        k = db2lin(-10.0)
        self.assert_(all(abs(level_0_1) <= k))

        ng.configure(level = -30)
        level_0_3 = ng(ones(1000))
        k = db2lin(-30.0)
        self.assert_(all(abs(level_0_3 - ones(1000)) <= k))

    def testEmpty(self):
        self.assertEqualVector(NoiseAdder()([]), [])

    def testNegatives(self):
        input = [1, -1, 2, -2, 3, -3]
        output = NoiseAdder(level = -10)(input)
        k = db2lin(-10.0)
        self.assert_( all(abs(output - input) <= k))

    def testFixSeed(self):
        a=NoiseAdder(fixSeed=True)(zeros(10))
        b=NoiseAdder(fixSeed=True)(zeros(10))
        self.assertEqualVector(a,b)


suite = allTests(TestNoiseAdder)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

