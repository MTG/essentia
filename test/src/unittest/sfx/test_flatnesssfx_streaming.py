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
from essentia.streaming import FlatnessSFX as sFlatnessSFX

class TestFlatnessSfx_Streaming(TestCase):

    def testRegression(self):
        # this algorithm has a standard mode implementation which has been
        # tested thru the unitests in python. Therefore it's only tested that
        # for a certain input standard == streaming
        envelope = range(22050)
        envelope.reverse()
        envelope = range(22050) + envelope

        # Calculate standard result
        stdResult = FlatnessSFX()(envelope)

        # Calculate streaming result
        p = Pool()
        input = VectorInput(envelope)
        accu = RealAccumulator()
        strFlatnessSfx = sFlatnessSFX()

        input.data >> accu.data
        accu.array >> strFlatnessSfx.envelope
        strFlatnessSfx.flatness >> (p, 'lowlevel.flatness')

        run(input)
        strResult = p['lowlevel.flatness']

        # compare results
        self.assertEqual(len(strResult), 1)
        self.assertAlmostEqual(strResult[0], stdResult, 5e-7)


suite = allTests(TestFlatnessSfx_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
