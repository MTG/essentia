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


class TestIIR(TestCase):

    def loadSimpleFilter(self):
        a = readVector(join(filedir(), 'filters/a.txt'))
        b = readVector(join(filedir(), 'filters/b.txt'))
        return IIR(numerator = b, denominator = a)


    def testInvalidParam(self):
        self.assertConfigureFails(IIR(), { 'numerator': [1, 2], 'denominator': [] })
        self.assertConfigureFails(IIR(), { 'numerator': [], 'denominator': [1, 2] })
        self.assertConfigureFails(IIR(), { 'numerator': [2, 3], 'denominator': [0, 2] })


    def testEmpty(self):
        filt = self.loadSimpleFilter()
        self.assertEqualVector(filt([]), [])


    def testOneByOne(self):
        # we compare here that filtering an array all at once or the samples
        # one by one will yield the same result
        filt = self.loadSimpleFilter()
        signal = readVector(join(filedir(), 'filters/x.txt'))

        expected = filt(signal)

        # need to reset the filter here!!
        filt.reset()

        result = []
        for sample in signal:
            result += list(filt([sample]))

        self.assertAlmostEqualVector(result, expected, 1e-6)


    def testZero(self):
        filt = self.loadSimpleFilter()
        self.assertEqualVector(filt(zeros(20)), zeros(20))

    def testRegression(self):
        # with len(a) == len(b)
        ba = readVector(join(filedir(), 'filters/ba.txt'))
        ab = readVector(join(filedir(), 'filters/ab.txt'))
        signal = readVector(join(filedir(), 'filters/x.txt'))
        expected = readVector(join(filedir(), 'filters/y1.txt'))

        filt = IIR(numerator = ba, denominator = ab)
        self.assertAlmostEqualVector(filt(signal), expected, 1e-6)

        # with len(b) > len(a)
        b = readVector(join(filedir(), 'filters/b.txt'))
        a = readVector(join(filedir(), 'filters/a.txt'))
        expected = readVector(join(filedir(), 'filters/y2.txt'))

        filt = IIR(numerator = b, denominator = a)
        self.assertAlmostEqualVector(filt(signal), expected, 1e-6)

        # with len(a) > len(b)
        expected = readVector(join(filedir(), 'filters/y3.txt'))

        filt = IIR(numerator = a, denominator = b)
        self.assertAlmostEqualVector(filt(signal), expected, 1e-6)






suite = allTests(TestIIR)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
