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
from essentia import *


class TestHFC(TestCase):

    def testRegression(self):
        # Simple regression test, comparing to reference values
        input = readVector(join(filedir(), 'highfrequencycontent/input.txt'))
        expected = readVector(join(filedir(), 'highfrequencycontent/output.txt'))
        self.assertAlmostEqual(HFC(sampleRate=44100, type='Masri')(input), expected[0], 1e-6)
        self.assertAlmostEqual(HFC(sampleRate=44100, type='Jensen')(input), expected[1], 1e-6)
        self.assertAlmostEqual(HFC(sampleRate=44100, type='Brossier')(input), expected[2], 1e-6)


    def testZero(self):
        # Inputting zeros should return zero hfc
        hfc = HFC(type='Masri')(zeros(1024))
        self.assertEqual(hfc, 0)

        hfc = HFC(type='Jensen')(zeros(1024))
        self.assertEqual(hfc, 0)

        hfc = HFC(type='Brossier')(zeros(1024))
        self.assertEqual(hfc, 0)

    def testInvalidParam(self):
        # Test different type than masri, jensen or brossier
        self.assertConfigureFails(HFC(), { 'type':'unknown'})

    def testEmpty(self):
        # Test that empty spectrum yields an exception and spectrum of size 1 returns
        # 0 hfc
        self.assertComputeFails(HFC(), [])

    def testOne(self):
        self.assertEqual(HFC()([9]), 0)


suite = allTests(TestHFC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
