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


class TestFrameBuffer(TestCase):

    def testEmpty(self):
        with self.assertRaises(RuntimeError):
            FrameBuffer()([])

    def testBufferZeroPadding(self):
        buffer = FrameBuffer(bufferSize=8, zeroPadding=True)
        self.assertEqualVector(buffer([1, 2]), [0., 0., 0., 0., 0., 0., 1., 2.])
        self.assertEqualVector(buffer([3, 4]), [0., 0., 0., 0., 1., 2., 3., 4.])
        self.assertEqualVector(buffer([5, 6]), [0., 0., 1., 2., 3., 4., 5., 6.])
        self.assertEqualVector(buffer([7, 8]), [1., 2., 3., 4., 5., 6., 7., 8.])
        self.assertEqualVector(buffer([9, 10]), [3., 4., 5., 6., 7.,  8., 9., 10.])

    def testBufferNoZeroPadding(self):
        buffer = FrameBuffer(bufferSize=8, zeroPadding=False)
        self.assertEqualVector(buffer([1, 2]), [])
        self.assertEqualVector(buffer([3, 4]), [])
        self.assertEqualVector(buffer([5, 6]), [])
        self.assertEqualVector(buffer([7, 8]), [1., 2., 3., 4., 5., 6., 7., 8.])

    def testFrameSizeEqualsBufferSize(self):
        buffer = FrameBuffer(bufferSize=8)
        self.assertEqualVector(buffer([1, 2, 3, 4, 5, 6, 7, 8]), [1., 2., 3., 4., 5., 6., 7., 8.])

    def testFrameSizeLargerBufferSize(self):
        buffer = FrameBuffer(bufferSize=8)
        self.assertEqualVector(buffer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3., 4., 5., 6., 7., 8., 9., 10.])

    def testResetZeroPadding(self):
        buffer = FrameBuffer(bufferSize=8, zeroPadding=True)
        buffer([1, 2, 3, 4, 5, 6])  # Results in [0., 0., 1., 2., 3., 4., 5., 6.]
        buffer.reset()  # Sets the buffer to zero vector.
        self.assertEqualVector(buffer([1, 2]), [0., 0., 0., 0., 0., 0., 1., 2.])

    def testResetNoZeroPadding(self):
        buffer = FrameBuffer(bufferSize=8, zeroPadding=False)
        buffer([1, 2, 3, 4, 5, 6, 7, 8])
        buffer.reset()
        self.assertEqualVector(buffer([1, 2]), [])


suite = allTests(TestFrameBuffer)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
