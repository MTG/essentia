#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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
import itertools

DIMS = 4  # Essentia only supports 4-dimensional Tensors.

class TestTensorTranspose(TestCase):

    def assertPermutations(self, original):
        # Iterate over all posible permutations
        indices = range(DIMS)
        for permutation in itertools.permutations(indices):
            expected = numpy.transpose(original, permutation)
            result = TensorTranspose(permutation=list(permutation))(original)

            self.assertEqualVector(result.shape, expected.shape)
            self.assertEqualVector(result.flatten(), expected.flatten())

    def testTranspose(self):
        length = 2
        original = numpy.arange(length ** DIMS, dtype='float32').reshape([length] * DIMS)
        self.assertPermutations(original)

    def testEmptyTensor(self):
        original = numpy.array([[[[]]]], dtype='float32')
        self.assertPermutations(original)

    def testUnitaryTensor(self):
        original = numpy.array([[[[1]]]], dtype='float32')
        self.assertPermutations(original)
    
    def testInvalidParam(self):
        self.assertConfigureFails(TensorTranspose(), { 'permutation': [0, -1, 2, 3] })
        self.assertConfigureFails(TensorTranspose(), { 'permutation': [0, 1, 2, 5] })
        self.assertConfigureFails(TensorTranspose(), { 'permutation': [0, 1, 2] })
        self.assertConfigureFails(TensorTranspose(), { 'permutation': [0, 1, 2, 2] })



suite = allTests(TestTensorTranspose)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
