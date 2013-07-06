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

class TestInstantPower(TestCase):

    def testZero(self):
        self.assertEqual(InstantPower()(zeros(1000)), 0)

    def testEmptyOrOne(self):
        self.assertComputeFails(InstantPower(), [])
        self.assertEqual(InstantPower()([23]), 23*23)


    def testRegression(self):
        input = [0, 1, 2, 3]
        expected = 14./4.

        self.assertEqual(InstantPower()(input), expected)



suite = allTests(TestInstantPower)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

