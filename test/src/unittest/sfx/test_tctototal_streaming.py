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
from essentia.streaming import TCToTotal as sTCToTotal

class TestTCToTotal(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        tcToTotal = sTCToTotal()
        p = Pool()

        gen.data >> tcToTotal.envelope
        tcToTotal.TCToTotal >> (p, 'lowlevel.tctototal')

        run(gen)

        self.assertRaises(KeyError, lambda: p['lowlevel.tctototal'])


    def testOneValue(self):
        gen = VectorInput([1])
        tcToTotal = sTCToTotal()
        p = Pool()

        gen.data >> tcToTotal.envelope
        tcToTotal.TCToTotal >> (p, 'lowlevel.tctototal')

        self.assertRaises(RuntimeError, lambda: run(gen))


    def testRegression(self):
        envelope = range(22050)
        envelope.reverse()
        envelope = range(22050) + envelope

        gen = VectorInput(envelope)
        tcToTotal = sTCToTotal()
        p = Pool()

        gen.data >> tcToTotal.envelope
        tcToTotal.TCToTotal >> (p, 'lowlevel.tctototal')

        run(gen)

        self.assertAlmostEqual(p['lowlevel.tctototal'],
                               TCToTotal()(envelope))


suite = allTests(TestTCToTotal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
