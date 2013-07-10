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

class TestLoudness(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(Loudness(), input)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(Loudness()(input), 0)

    def testOne(self):
        input = [0]
        self.assertAlmostEqual(Loudness()(input), 0)

        input = [100]
        self.assertAlmostEqual(Loudness()(input), 478.63009232263852, 1e-6)

    def testRegression(self):
        input = [45, 78, 1, -5, -.1125, 1.236, 10.25, 100, 9, -78]
        self.assertAlmostEqual(Loudness()(input), 870.22171051882947, 1e-6)


suite = allTests(TestLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
