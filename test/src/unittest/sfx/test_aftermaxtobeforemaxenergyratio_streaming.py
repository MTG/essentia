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
from essentia.streaming import AfterMaxToBeforeMaxEnergyRatio as \
        sAfterMaxToBeforeMaxEnergyRatio

class TestAfterMaxToBeforeMaxEnergyRatio_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        strRatio = sAfterMaxToBeforeMaxEnergyRatio()
        p = Pool()

        gen.data >> strRatio.pitch
        strRatio.afterMaxToBeforeMaxEnergyRatio >> (p, 'lowlevel.amtbmer')

        run(gen)

        self.assertRaises(KeyError, lambda: p['lowlevel.amtbmer'])


    def testRegression(self):
        # this algorithm has a standard mode implementation which has been
        # tested thru the unitests in python. Therefore it's only tested that
        # for a certain input standard == streaming
        pitch = readVector(join(filedir(), 'aftermaxtobeforemaxenergyratio', 'input.txt'))

        p = Pool()
        gen = VectorInput(pitch)
        strRatio = sAfterMaxToBeforeMaxEnergyRatio()

        gen.data >> strRatio.pitch
        strRatio.afterMaxToBeforeMaxEnergyRatio >> (p, 'lowlevel.amtbmer')

        run(gen)

        stdResult = AfterMaxToBeforeMaxEnergyRatio()(pitch)
        strResult = p['lowlevel.amtbmer']
        self.assertAlmostEqual(strResult, stdResult, 5e-7)


suite = allTests(TestAfterMaxToBeforeMaxEnergyRatio_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
