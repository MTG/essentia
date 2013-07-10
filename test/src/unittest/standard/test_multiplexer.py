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

class TestMultiplexer(TestCase):

    def computeRealInput(self, N, M):
        inputs = []
        expected = []
        for n in range(N):
            inputs.append([float(i) for i in range(n*M, (n+1)*M)])

        for m in range(M):
            expected.append([i for i in range(m, N*M, M)])
        return inputs, expected


    def computeVectorInput(self, nVectors, N, M):
        expected = zeros([N,  M*nVectors])
        inputs = [[[float(m + n*M + i*N*M) for m in range(M)] for n in range(N)] for i in range(nVectors)]

        for n in range(N):
            for i in range(nVectors):
                for m in range(M):
                    expected[n][m+i*M] = inputs[i][n][m]

        return inputs, expected

    def computeRealVectorInput(self, nVectors, N, M):

        #### Reals ###
        inputReals = []
        expectedReals = []
        #for n in range(N):
        #    inputReals.append([float(i) for i in range(n*M, (n+1)*M)])
        inputReals = [[float(i) for i in range(n*N,(n+1)*N)] for n in range(M)]

        for m in range(M):
            expectedReals.append(range(m, N*M, M))


        #### Vectors ###

        inputVectors = [[[float(m + n*M + i*N*M) for m in range(M)] for n in range(N)] for i in range(nVectors)]



        expectedVectors = zeros([N,  M*nVectors])
        for n in range(N):
            for i in range(nVectors):
                for m in range(M):
                    expectedVectors[n][m+i*M] = inputVectors[i][n][m]

        ### expected result ###
        expected = [[]]*(len(expectedReals) + len(expectedVectors))
        for i in range(len(expectedReals)): expected[i] = expectedReals[i] #exp in expectedReals: expected.append(exp)
        for i in range(len(expectedVectors)):
            for exp in expectedVectors[i]:
                expected[i].append(exp)

        return inputReals, inputVectors, expected





    def testReal(self):
        (inputs, expected) = self.computeRealInput(6, 15)
        mux = Multiplexer(numberRealInputs=6)
        result = mux(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])

        self.assertEqualMatrix(result, expected)

    def testRealEmpty(self):
        (inputs, expected) = self.computeRealInput(2, 0)
        mux = Multiplexer(numberRealInputs=2)
        result = mux(inputs[0], inputs[1])
        self.assertEqualVector(result, expected)

    def testVector(self):
        (inputs, expected) = self.computeVectorInput(6, 15, 4)
        mux = Multiplexer(numberVectorRealInputs=6)
        result = mux(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
        self.assertEqualMatrix(result, expected)

    def testVectorEmpty(self):
        (inputs, expected) = self.computeVectorInput(6, 15, 0)
        mux = Multiplexer(numberVectorRealInputs=6)
        result = mux(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
        self.assertEqualMatrix(result, expected)

    def testRealVector(self):
        inputReals, inputVectors, expected = self.computeRealVectorInput(6, 15, 3)
        mux = Multiplexer(numberRealInputs=3, numberVectorRealInputs=6)
        result = mux(inputReals[0], inputReals[1], inputReals[2],
                     inputVectors[0], inputVectors[1], inputVectors[2],inputVectors[3], inputVectors[4], inputVectors[5])


suite = allTests(TestMultiplexer)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
