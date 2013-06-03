#!/usr/bin/env python

from essentia_test import *
from numpy import dot # dot product

testdir = join(filedir(), 'singlegaussian')


class TestSingleGaussian(TestCase):

    def assertInverse(self, cov, icov):
        (rows,cols) = cov.shape
        self.assertEqual(rows, cols)
        I = zeros([rows,cols]) # identity matrix
        for i in xrange(rows): I[i][i] = 1.0
        # assert that covariance*inverse_covariance = identity matrix
        self.assertAlmostEqualMatrix(dot(cov,icov), I, 1e-6)

    def testRegression(self):
        input = readMatrix(join(testdir, 'matrix.txt'))
        expectedMean = readVector(join(testdir, 'mean.txt'))
        expectedCov = readMatrix(join(testdir, 'cov.txt'))
        expectedInv = readMatrix(join(testdir, 'invcov.txt'))

        (outputMean, outputCov, outputInv) = SingleGaussian()(input)

        self.assertAlmostEqualVector(outputMean, expectedMean, 1e-6)
        self.assertAlmostEqualMatrix(outputCov, expectedCov, 1e-6)
        self.assertAlmostEqualMatrix(outputInv, expectedInv, 1e-6)
        self.assertInverse(outputCov, outputInv)


    def testRegressionStreaming(self):
        from essentia.streaming import SingleGaussian as strSingleGaussian
        input = [readMatrix(join(testdir, 'matrix.txt'))]
        expectedMean = readVector(join(testdir, 'mean.txt'))
        expectedCov = readMatrix(join(testdir, 'cov.txt'))
        expectedInv = readMatrix(join(testdir, 'invcov.txt'))

        gen = VectorInput(input)
        singleGaussian = strSingleGaussian()
        pool = Pool()

        gen.data >> singleGaussian.matrix
        singleGaussian.mean >> (pool, 'mean')
        singleGaussian.covariance >> (pool, 'covariance')
        singleGaussian.inverseCovariance >> (pool, 'invCovariance')
        run(gen)

        self.assertAlmostEqualVector(pool['mean'][0], expectedMean, 1e-6)
        self.assertAlmostEqualMatrix(pool['covariance'][0], expectedCov, 1e-6)
        self.assertAlmostEqualMatrix(pool['invCovariance'][0], expectedInv, 1e-6)
        self.assertInverse(pool['covariance'][0], pool['invCovariance'][0])

    def testZero(self):
        self.assertComputeFails(SingleGaussian(), array([[0]]))

    def testEmpty(self):
        self.assertComputeFails(SingleGaussian(),array([[]]))

    def testOneRow(self):
        mat = array([[1,2,3,4,5]])
        self.assertComputeFails(SingleGaussian(), mat)

    def testOneCol(self):
        mat = array([[1],[2],[3],[4],[5]])
        mean = array(3)
        cov =  array([[ 2.5]])
        icov = array([[ 0.4]])
        outputMean, outputCov, outputInv = SingleGaussian()(mat)
        self.assertEqual(outputMean, mean)
        self.assertAlmostEqualMatrix(outputCov, cov)
        self.assertAlmostEqualMatrix(outputInv, icov)
        self.assertInverse(outputCov, outputInv)



suite = allTests(TestSingleGaussian)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
