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
from essentia.streaming import RealAccumulator

class TestRealAccumulator_Streaming(TestCase):

    def accumulate(self, input, size = 1):
        # NB: size is not used anymore as RealAccumulator got smarter :-)
        accu = RealAccumulator()
        pool = Pool()
        gen = VectorInput(input)
        gen.data >> accu.data
        accu.array >> (pool, 'accu')
        run(gen)
        if not pool.descriptorNames() : return []
        return pool['accu']

    def testEmpty(self):
        self.assertEqualVector(self.accumulate([]), [])

    def testSingle(self):
        self.assertEqual(self.accumulate([1.0]), [1.0])

    def testRegression(self):
        input = [5.0, 0.0, -1.0, 2.0, -3.0, 4.0];
        result = self.accumulate(input)
        self.assertEqualVector(result, input)

    def testPreferredSize(self):
        # NB: this is not as useful as before when we had a preferredSize parameter
        input = [ float(i) for i in xrange(44100)]
        result = self.accumulate(input, size=1024)
        self.assertEqualVector(result, input)

suite = allTests(TestRealAccumulator_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
