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
from essentia.streaming import Derivative

class TestDerivative_Streaming(TestCase):

    def testRegression(self):
        input = [5.0, 0.0, -1.0, 2.0, -3.0, 4.0]
        expected = [5.0, -5.0, -1.0, 3.0, -5.0, 7.0]

        gen = VectorInput(input)
        der = Derivative()
        p = Pool()

        gen.data >> der.signal
        der.signal >> (p, 'data')
        run(gen)

        self.assertEqualVector(p['data'], expected)

    def testEmpty(self):
        gen = VectorInput([])
        der = Derivative()
        p = Pool()

        gen.data >> der.signal
        der.signal >> (p, 'data')
        run(gen)

        self.assertEqualVector(p.descriptorNames(), [])

    def testSingle(self):
        gen = VectorInput([])
        der = Derivative()
        p = Pool()

        gen.data >> der.signal
        der.signal >> (p, 'data')
        run(gen)

        self.assertEqualVector(p.descriptorNames(), [])

    def testStdVsStreaming(self):
        from essentia.standard import Derivative as stdDerivative
        input = [5.0, 0.0, -1.0, 2.0, -3.0, 4.0]
        expected = [5.0, -5.0, -1.0, 3.0, -5.0, 7.0]
        self.assertEqualVector(stdDerivative()(input), expected)


suite = allTests(TestDerivative_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
