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
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TestTensorNormalize(TestCase):

    def scalerSingleValue(self, scaler_arg=('standard', StandardScaler())):
        scaler_name, scaler = scaler_arg

        original = numpy.arange(1, dtype='float32')

        # Use Scipy to generate the expected results
        expected =scaler.fit_transform(original.reshape(-1, 1))

        # Add singleton dimensions to trainsform the input vector into a tensor
        original = numpy.expand_dims(original, axis=[0, 1, 2])
        result = TensorNormalize(scaler=scaler_name, axis=-1)(original)

        self.assertAlmostEqualVector(result.flatten(), expected.flatten(), 1e-6)

    def scalerOverall(self, scaler_arg=('standard', StandardScaler())):
        scaler_name, scaler = scaler_arg

        original = numpy.arange(3, dtype='float32')

        expected = scaler.fit_transform(original.reshape(-1, 1))

        original = numpy.expand_dims(original, axis=[0, 1, 2])
        result = TensorNormalize(scaler=scaler_name, axis=-1)(original)

        self.assertAlmostEqualVector(result.flatten(), expected.flatten(), 1e-6)

    def scalerAlongAxis(self, axis=0, scaler_arg=('standard', StandardScaler())):
        scaler_name, scaler = scaler_arg

        dims, length = 4, 2
        original = numpy.arange(length ** dims, dtype='float32').reshape([length] * dims)

        expected = numpy.empty(original.shape)

        for i in range(original.shape[0]):
            # Swap the axes before and after so we can test all the dimensions
            # using the same operator ([i, :, :, :]).
            original = numpy.swapaxes(original, 0, axis)
            expected = numpy.swapaxes(expected, 0, axis)
            tmp = scaler.fit_transform(original[i, :, :, :].reshape(-1, 1))
            expected[i, :, :, :] = tmp.reshape([1] + [length] * (dims - 1))
            original = numpy.swapaxes(original, 0, axis)
            expected = numpy.swapaxes(expected, 0, axis)

        result = TensorNormalize(scaler=scaler_name, axis=axis)(original)
        self.assertAlmostEqualVector(result.flatten(), expected.flatten(), 1e-6)

    def scalerOverallConstantValue(self, scaler_arg=('standard', StandardScaler())):
        scaler_name, scaler = scaler_arg

        original = numpy.ones([2, 2], dtype='float32')

        # Use Scipy to generate the expected results
        expected =scaler.fit_transform(original.reshape(-1, 1))

        # Add singleton dimensions to trainsform the input vector into a tensor
        original = numpy.expand_dims(original, axis=[0, 1])
        result = TensorNormalize(scaler=scaler_name, axis=-1)(original)

        self.assertAlmostEqualVector(result.flatten(), expected.flatten(), 1e-6)

    def scalerAlongAxisConstantValue(self, axis=0, scaler_arg=('standard', StandardScaler())):
        scaler_name, scaler = scaler_arg

        dims, length = 4, 2
        original = numpy.arange(length ** dims, dtype='float32').reshape([length] * dims)

        # Constant along in one of the axes
        original[0,:,:,:] = numpy.ones(length ** (dims - 1), dtype='float32').reshape([length] * (dims - 1))

        expected = numpy.empty(original.shape)

        for i in range(original.shape[0]):
            # Swap the axes before and after so we can test all the dimensions
            # using the same operator ([i, :, :, :]).
            original = numpy.swapaxes(original, 0, axis)
            expected = numpy.swapaxes(expected, 0, axis)
            tmp = scaler.fit_transform(original[i, :, :, :].reshape(-1, 1))
            expected[i, :, :, :] = tmp.reshape([1] + [length] * (dims - 1))
            original = numpy.swapaxes(original, 0, axis)
            expected = numpy.swapaxes(expected, 0, axis)

        result = TensorNormalize(scaler=scaler_name, axis=axis)(original)
        self.assertAlmostEqualVector(result.flatten(), expected.flatten(), 1e-6)

    def testStandardScalerOverall(self):
        self.scalerOverall(scaler_arg=('standard', StandardScaler()))

    def testMinMaxScalerOverall(self):
        self.scalerOverall(scaler_arg=('minMax', MinMaxScaler()))

    def testStandardScalerAlongAxis(self):
        for i in range(4):
            self.scalerAlongAxis(axis=i, scaler_arg=('standard', StandardScaler()))

    def testMinMaxScalerAlongAxis(self):
        for i in range(4):
            self.scalerAlongAxis(axis=i, scaler_arg=('minMax', MinMaxScaler()))

    def testStandrdScalerOverallConstantValue(self):
        self.scalerOverallConstantValue(scaler_arg=('standard', StandardScaler()))

    def testMinMaxScalerOverallConstantValue(self):
        self.scalerOverallConstantValue(scaler_arg=('minMax', MinMaxScaler()))

    def testStandardScalerAlongAxisConstantValue(self):
        for i in range(4):
            self.scalerAlongAxisConstantValue(axis=i, scaler_arg=('standard', StandardScaler()))

    def testMinMaxScalerAlongAxisConstantValue(self):
        for i in range(4):
            self.scalerAlongAxisConstantValue(axis=i, scaler_arg=('minMax', MinMaxScaler()))

    def testInvalidParam(self):
        self.assertConfigureFails(TensorNormalize(), { 'axis': -2 })
        self.assertConfigureFails(TensorNormalize(), { 'axis': 5 })
        self.assertConfigureFails(TensorNormalize(), { 'scaler': 'MAXMIN' })



suite = allTests(TestTensorNormalize)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
