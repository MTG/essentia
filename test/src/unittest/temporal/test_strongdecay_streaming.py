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
from essentia.streaming import StrongDecay as sStrongDecay

class TestStrongDecay_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        strongDecay = sStrongDecay()
        p = Pool()

        gen.data >> strongDecay.signal
        strongDecay.strongDecay >> (p, 'strongDecay')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testOneValue(self):
        gen = VectorInput([1.0])
        strongDecay = sStrongDecay()
        p = Pool()

        gen.data >> strongDecay.signal
        strongDecay.strongDecay >> (p, 'strongDecay')

        self.assertRaises(EssentiaException, lambda: run(gen))


    def testRegression(self):
        # borrowing lpc's input vector for this regression test
        input = readVector(join(filedir(), 'lpc', 'input.txt'))

        gen = VectorInput(input)
        strongDecay = sStrongDecay()
        p = Pool()

        gen.data >> strongDecay.signal
        strongDecay.strongDecay >> (p, 'strongDecay')

        run(gen)

        self.assertAlmostEqual(p['strongDecay'], StrongDecay()(input), 1e-6)


suite = allTests(TestStrongDecay_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
