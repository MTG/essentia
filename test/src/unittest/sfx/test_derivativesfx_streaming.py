#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import DerivativeSFX as sDerivativeSfx

class TestDerivativeSfx_Streaming(TestCase):

    def helper(self, input):
        gen = VectorInput(input)
        accu = RealAccumulator()
        derivativeSfx = sDerivativeSfx()
        p = Pool()

        gen.data >> accu.data
        accu.array >> derivativeSfx.envelope
        derivativeSfx.derAvAfterMax >> (p, 'sfx.derAvAfterMax')
        derivativeSfx.maxDerBeforeMax >> (p, 'sfx.maxDerBeforeMax')

        run(gen)

        return (p['sfx.derAvAfterMax'][0],
                p['sfx.maxDerBeforeMax'][0])

    def testZero(self):
        output = self.helper([0]*100)

        self.assertEqual(output[0], 0)
        self.assertEqual(output[1], 0)

    def testOne(self):
        output = self.helper([1234.0])

        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 1234.)

    def testAscending(self):
        output = self.helper( [x/99. for x in range(100)] )

        self.assertEqual(output[0], output[1])

    def testDescending(self):
        input = [x/99. for x in range(100)]
        input.reverse()

        output = self.helper(input)

        self.assertEqual(output[0], 0.)
        self.assertEqual(output[1], 1.)

    def testRegression(self):
        input = [x/99. for x in range(100)]
        input.reverse()

        input = [x/99. for x in range(100)] + input

        output = self.helper(input)

        self.assertAlmostEqual(output[0], -0.0194097850471735, 1e-6)
        self.assertAlmostEqual(output[1], 0.010101020336151123, 1e-6)

suite = allTests(TestDerivativeSfx_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
