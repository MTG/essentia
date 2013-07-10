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
from essentia.streaming import Leq as sLeq
from math import sin, pi

class TestLeq_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        leq = sLeq()
        p = Pool()

        gen.data >> leq.signal
        leq.leq >> (p, 'leq')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testRegression(self):
        input = [1. / (i+1) * sin(2*pi*440*i/44100) for i in range(22050)]
        input += [0]*22050

        gen = VectorInput(input)
        leq = sLeq()
        p = Pool()

        gen.data >> leq.signal
        leq.leq >> (p, 'leq')

        run(gen)

        self.assertAlmostEqual(p['leq'], Leq()(input), 5e-5)


suite = allTests(TestLeq_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
