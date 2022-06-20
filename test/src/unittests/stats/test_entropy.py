#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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


class TestEntropy(TestCase):

    def testZero(self):
        self.assertEqual(Entropy()(zeros(256)), 0)

    def testOnes(self):
        # Test for different powers of 2
        self.assertEqual(Entropy()(ones(16)), 4)
        self.assertEqual(Entropy()(ones(32)), 5)
        self.assertEqual(Entropy()(ones(64)), 6)

    def testEmpty(self):
        self.assertRaises(EssentiaException, lambda: Entropy()([]))

    def testInvalidInput(self):
        input = [10, 10 , 10, 10, -10]
        self.assertRaises(RuntimeError, lambda: Entropy()(input))

    def testRegression(self):
        # Check value with online calculator
        output = Entropy()([0.9, 0.8, 0.7, 0.6])
        #https://planetcalc.com/2476/
        expected_output = 1.98387111 # from above calculator, 8 decimal places
        self.assertAlmostEqual(expected_output, output, 8)

        output = Entropy()([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6])
        #https://planetcalc.com/2476/
        expected_output = 1.0+1.98387111 # from above calculator, 8 decimal places
        self.assertAlmostEqual(expected_output, output, 8)



suite = allTests(TestEntropy)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
