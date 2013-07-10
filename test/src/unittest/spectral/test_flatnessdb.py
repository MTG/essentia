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

class TestFlatnessDB(TestCase):

    def testEmpty(self):
        input = []
        self.assertComputeFails(FlatnessDB(), input)

    def testZeros(self):
        input = [0]*100
        result = FlatnessDB()(input)
        self.assertEqual(result, 1.0)

    def testFib(self):
        input = [1,1,2,3,5,8,13,21,34]
        result = FlatnessDB()(input)

        # calculated by hand (jk, with a calculator)
        expected = 0.047487197381536603376080461923790

        self.assertAlmostEqual(result, expected)


suite = allTests(TestFlatnessDB)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
