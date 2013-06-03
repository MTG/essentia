#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import MaxToTotal as sMaxToTotal

class TestMaxToTotal_Streaming(TestCase):

    def testRegression(self):
        # triangle input
        envelope = range(22050)
        envelope.reverse()
        envelope = range(22050) + envelope

        gen = VectorInput(envelope)
        maxToTotal = sMaxToTotal()
        p = Pool()

        gen.data >> maxToTotal.envelope
        maxToTotal.maxToTotal >> (p, 'lowlevel.maxToTotal')

        run(gen)

        result = p['lowlevel.maxToTotal']
        self.assertAlmostEqual(result, .5, 5e-5) #this seems like a large error -rtoscano
        self.assertAlmostEqual(result, MaxToTotal()(envelope), 5e-7)

    def testEmpty(self):
        gen = VectorInput([])
        alg = sMaxToTotal()
        p = Pool()

        gen.data >> alg.envelope
        alg.maxToTotal >> (p, 'lowlevel.maxToTotal')

        run(gen)

        # Make sure nothing was emitted to the pool
        self.assertRaises(KeyError, lambda: p['lowlevel.maxToTotal'])


suite = allTests(TestMaxToTotal_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
