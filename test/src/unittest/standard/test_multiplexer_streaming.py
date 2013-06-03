#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import Multiplexer

class TestMultiplexer_Streaming(TestCase):

    def realTest(self, N, M):
        inputs = []
        expected = []
        for n in range(N):
            inputs.append([float(i) for i in range(n*M, (n+1)*M)])

        for m in range(M):
            expected.append(range(m, N*M, M))

        gens = [VectorInput(input) for input in inputs]
        mux = Multiplexer(numberRealInputs=N)
        ports = []
        for i in range(N):
            ports.append(getattr(mux, "real_" + str(i)))
            gens[i].data >> ports[i]

        pool = Pool()
        mux.data >> (pool, "real")
        for gen in gens: run(gen)

        if not N or not M:
            self.assertEqual(pool.descriptorNames(), [])
            return

        self.assertEqualMatrix(pool['real'], expected)

    def vectorTest(self, nVectors, N, M):
        inputs = zeros([nVectors, N, M])
        expected = zeros([N,  M*nVectors])

        for i in range(nVectors):
            for n in range(N):
                for m in range(M):
                    inputs[i][n][m] = float(m + n*M + i*N*M)

        for n in range(N):
            for i in range(nVectors):
                for m in range(M):
                    expected[n][m+i*M] = inputs[i][n][m]
        gens = [VectorInput(array(input)) for input in inputs]
        mux = Multiplexer(numberVectorRealInputs=nVectors)
        ports = []
        for i in range(nVectors):
            ports.append(getattr(mux, "vector_" + str(i)))
            gens[i].data >> ports[i]

        pool = Pool()
        mux.data >> (pool, "vector")
        for gen in gens: run(gen)

        if not N or not nVectors:
            self.assertEqual(pool.descriptorNames(), [])
            return

        if not M : expected = [[],[],[],[],[]]

        self.assertEqualMatrix(pool['vector'], expected)

    def realVectorTest(self, nVectors, N, M):

        #### Reals ###
        inputReals = []
        expectedReals = []
        for n in range(N):
            inputReals.append([float(i) for i in range(n*M, (n+1)*M)])

        for m in range(M):
            expectedReals.append(range(m, N*M, M))

        genReals = [VectorInput(input) for input in inputReals]

        #### Vectors ###

        inputVectors = zeros([nVectors, N, M])
        expectedVectors = zeros([N,  M*nVectors])

        for i in range(nVectors):
            for n in range(N):
                for m in range(M):
                    inputVectors[i][n][m] = m + n*M + i*N*M

        for n in range(N):
            for i in range(nVectors):
                for m in range(M):
                    expectedVectors[n][m+i*M] = inputVectors[i][n][m]
        genVectors = [VectorInput(array(input)) for input in inputVectors]

        ### expected result ###
        expected = []
        for exp in expectedReals: expected.append(exp)
        for i in range(len(expectedVectors)):
            for exp in expectedVectors[i]:
                expected[i].append(exp)

        ### network ###
        mux = Multiplexer(numberRealInputs=N,
                          numberVectorRealInputs=nVectors)
        portReals = []
        for i in range(N):
            portReals.append(getattr(mux, "real_" + str(i)))
            genReals[i].data >> portReals[i]

        portVectors = []
        for i in range(nVectors):
            portVectors.append(getattr(mux, "vector_" + str(i)))
            genVectors[i].data >> portVectors[i]

        pool = Pool()
        mux.data >> (pool, "realvector")
        for gen in genReals: run(gen)
        for gen in genVectors: run(gen)

        if not N or not nVectors:
            self.assertEqual(pool.descriptorNames(), [])
            return

        if not M : expected = [[],[],[],[],[]]

        self.assertEqualMatrix(pool['realvector'], expected)

    def testReal(self):
        self.realTest(1,1)
        self.realTest(3,5)
        self.realTest(3,3)

    def testRealEmpty(self):
        self.realTest(3,0)
        self.realTest(0,0)
        self.realTest(0,3)

    def testVector(self):
        self.vectorTest(3,6,5)
        self.vectorTest(1,1,1)
        self.vectorTest(3,3,3)

    def testVectorEmpty(self):
        self.vectorTest(3,0,5)
        self.vectorTest(0,6,5)
        self.vectorTest(3,5,0)

    def testRealVector(self):
        self.realVectorTest(3,3,3)


suite = allTests(TestMultiplexer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
