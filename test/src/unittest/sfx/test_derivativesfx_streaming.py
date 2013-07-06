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
from essentia.streaming import DerivativeSFX as sDerivativeSfx

class TestDerivativeSfx_Streaming(TestCase):

    def helper(self, input):
        gen = VectorInput(input)
        accu = RealAccumulator()
        derivativeSfx = sDerivativeSfx()
        p = Pool()

        gen.data >> accu.data
        accu.array >> derivativeSfx.envelope
        derivativeSfx.derAvAfterMax >> (p, 'sfx.derAvAfterMax')
        derivativeSfx.maxDerBeforeMax >> (p, 'sfx.maxDerBeforeMax')

        run(gen)

        return (p['sfx.derAvAfterMax'][0],
                p['sfx.maxDerBeforeMax'][0])

    def testZero(self):
        output = self.helper([0]*100)

        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 0)

    def testOne(self):
        output = self.helper([1234.0])

        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 1234.)

    def testAscending(self):
        output = self.helper( [x/99. for x in range(100)] )

        self.assertEqual(output[0], output[1])

    def testDescending(self):
        input = [x/99. for x in range(100)]
        input.reverse()

        output = self.helper(input)

        self.assertEqual(output[0], 0.)
        self.assertEqual(output[1], 1.)

    def testRegression(self):
        input = [x/99. for x in range(100)]
        input.reverse()

        input = [x/99. for x in range(100)] + input

        output = self.helper(input)

        self.assertAlmostEqual(output[0], -0.0194097850471735, 1e-6)
        self.assertAlmostEqual(output[1], 0.010101020336151123, 1e-6)

suite = allTests(TestDerivativeSfx_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
