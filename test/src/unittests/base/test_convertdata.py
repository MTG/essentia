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
from essentia.common import Edt, determineEdt, convertData
from essentia.standard import MinMax
import numpy


class TestConvertData(TestCase):
    """Regression coverage for determineEdt()/convertData() accepting a
    Python list whose elements are numpy.float32 scalars (Edt.NUMPY_FLOAT),
    not just a list of plain Python floats (Edt.REAL).

    Indexing an essentia VECTOR_REAL result (e.g. HPCP output) yields
    numpy.float32 elements, so accumulating per-frame output into a plain
    Python list -- the ordinary way to do it, as in Key's test_key.py
    runAlg() or HPCP's own test_hpcp.py testWhiteNoise() -- produces
    exactly this kind of list. Before this fix, determineEdt() fell
    through its whole dispatch table for such a list and returned
    Edt.UNDEFINED, so passing it back into any algorithm expecting
    VECTOR_REAL raised: TypeError: Cannot convert data from type UNDEFINED
    (<class 'list'>) to type VECTOR_REAL.
    """

    def testListOfNumpyFloat32DeterminesListReal(self):
        values = [numpy.float32(0.5), numpy.float32(1.5)]
        self.assertEqual(determineEdt(values), Edt.LIST_REAL)

    def testListOfNumpyFloat32ConvertsToVectorReal(self):
        values = [numpy.float32(0.5), numpy.float32(1.5)]
        converted = convertData(values, Edt.VECTOR_REAL)
        self.assertEqualVector(list(converted), [0.5, 1.5])

    def testListOfNumpyFloat32AcceptedByAlgorithm(self):
        """End-to-end: MinMax() takes a VECTOR_REAL argument, exactly the
        conversion path test_key.py/test_hpcp.py hit."""
        values = [numpy.float32(0.5), numpy.float32(1.5), numpy.float32(-2.0)]
        value, index = MinMax()(values)
        self.assertEqual(value, -2.0)
        self.assertEqual(index, 2)

    def testAccumulatedFrameOutputAcceptedByAlgorithm(self):
        """Reproduces the accumulate-per-frame pattern directly: summing
        numpy.float32 elements with += promotes a plain 0 to numpy.float32,
        and a division by a Python int keeps it numpy.float32, so the
        resulting list is exactly the Edt.NUMPY_FLOAT case."""
        frames = [
            numpy.array([1.0, 2.0], dtype='f4'),
            numpy.array([3.0, 4.0], dtype='f4'),
        ]
        sums = [0, 0]
        for frame in frames:
            for i in range(len(frame)):
                sums[i] += frame[i]
        averaged = [x / len(frames) for x in sums]

        self.assertTrue(all(isinstance(x, numpy.float32) for x in averaged))
        value, index = MinMax()(averaged)
        self.assertEqual(value, 2.0)
        self.assertEqual(index, 0)


suite = allTests(TestConvertData)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
