#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import MinToTotal as sMinToTotal

class TestMinToTotal_Streaming(TestCase):

    def testRegression(self):
        # V-shaped input
        envelope = range(22050)
        envelope.reverse()
        envelope += range(22050)

        gen = VectorInput(envelope)
        minToTotal = sMinToTotal()
        p = Pool()

        gen.data >> minToTotal.envelope
        minToTotal.minToTotal >> (p, 'lowlevel.minToTotal')

        run(gen)

        result = p['lowlevel.minToTotal']
        self.assertAlmostEqual(result, .5, 5e-5) #this seems like a large error -rtoscano
        self.assertAlmostEqual(result, MinToTotal()(envelope), 5e-7)

    def testEmpty(self):
        gen = VectorInput([])
        alg = sMinToTotal()
        p = Pool()

        gen.data >> alg.envelope
        alg.minToTotal >> (p, 'lowlevel.minToTotal')

        run(gen)

        # Make sure nothing was emitted to the pool
        self.assertRaises(KeyError, lambda: p['lowlevel.minToTotal'])


suite = allTests(TestMinToTotal_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
