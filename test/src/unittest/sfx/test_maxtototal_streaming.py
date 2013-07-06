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
from essentia.streaming import MaxToTotal as sMaxToTotal

class TestMaxToTotal_Streaming(TestCase):

    def testRegression(self):
        # triangle input
        envelope = range(22050)
        envelope.reverse()
        envelope = range(22050) + envelope

        gen = VectorInput(envelope)
        maxToTotal = sMaxToTotal()
        p = Pool()

        gen.data >> maxToTotal.envelope
        maxToTotal.maxToTotal >> (p, 'lowlevel.maxToTotal')

        run(gen)

        result = p['lowlevel.maxToTotal']
        self.assertAlmostEqual(result, .5, 5e-5) #this seems like a large error -rtoscano
        self.assertAlmostEqual(result, MaxToTotal()(envelope), 5e-7)

    def testEmpty(self):
        gen = VectorInput([])
        alg = sMaxToTotal()
        p = Pool()

        gen.data >> alg.envelope
        alg.maxToTotal >> (p, 'lowlevel.maxToTotal')

        run(gen)

        # Make sure nothing was emitted to the pool
        self.assertRaises(KeyError, lambda: p['lowlevel.maxToTotal'])


suite = allTests(TestMaxToTotal_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
