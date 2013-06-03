#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import Derivative

class TestDerivative_Streaming(TestCase):

    def testRegression(self):
        input = [5.0, 0.0, -1.0, 2.0, -3.0, 4.0]
        expected = [5.0, -5.0, -1.0, 3.0, -5.0, 7.0]

        gen = VectorInput(input)
        der = Derivative()
        p = Pool()

        gen.data >> der.signal
        der.signal >> (p, 'data')
        run(gen)

        self.assertEqualVector(p['data'], expected)

    def testEmpty(self):
        gen = VectorInput([])
        der = Derivative()
        p = Pool()

        gen.data >> der.signal
        der.signal >> (p, 'data')
        run(gen)

        self.assertEqualVector(p.descriptorNames(), [])

    def testSingle(self):
        gen = VectorInput([])
        der = Derivative()
        p = Pool()

        gen.data >> der.signal
        der.signal >> (p, 'data')
        run(gen)

        self.assertEqualVector(p.descriptorNames(), [])

    def testStdVsStreaming(self):
        from essentia.standard import Derivative as stdDerivative
        input = [5.0, 0.0, -1.0, 2.0, -3.0, 4.0]
        expected = [5.0, -5.0, -1.0, 3.0, -5.0, 7.0]
        self.assertEqualVector(stdDerivative()(input), expected)


suite = allTests(TestDerivative_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
