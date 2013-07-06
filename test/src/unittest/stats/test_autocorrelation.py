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

testdir = join(filedir(), 'autocorrelation')


class TestAutoCorrelation(TestCase):

    def testRegression(self):
        inputv = readVector(join(testdir, 'input_pow2.txt'))
        expected = readVector(join(testdir, 'output.txt'))

        output = AutoCorrelation()(inputv)

        self.assertAlmostEqualVector(expected, output, 1e-4)


    def testNonPowerOfTwo(self):
        inputv = readVector(join(testdir, 'octave_input.txt'))
        inputv = inputv[:234]
        expected = readVector(join(testdir, 'output_nonpow2.txt'))

        output = AutoCorrelation()(inputv)

        self.assertAlmostEqualVector(expected, output, 1e-4)


    def testOctave(self):
        inputv = readVector(join(testdir, 'octave_input.txt'))
        expected = readVector(join(testdir, 'octave_output.txt'))

        output = AutoCorrelation()(inputv)

        self.assertEqual(len(expected)/2, len(output))

        self.assertAlmostEqualVector(expected[:len(expected)/2], output, 1e-4)


    def testZero(self):
        self.assertEqualVector(AutoCorrelation()(zeros(1024)), zeros(1024))

    def testEmpty(self):
        self.assertEqualVector(AutoCorrelation()([]), [])

    def testOne(self):
        self.assertAlmostEqualVector(AutoCorrelation()([0.2]), [0.04])

    def testInvalidParam(self):
        self.assertConfigureFails(AutoCorrelation(), { 'normalization': 'unknown' })


suite = allTests(TestAutoCorrelation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
