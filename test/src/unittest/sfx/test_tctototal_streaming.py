#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import TCToTotal as sTCToTotal

class TestTCToTotal(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        tcToTotal = sTCToTotal()
        p = Pool()

        gen.data >> tcToTotal.envelope
        tcToTotal.TCToTotal >> (p, 'lowlevel.tctototal')

        run(gen)

        self.assertRaises(KeyError, lambda: p['lowlevel.tctototal'])


    def testOneValue(self):
        gen = VectorInput([1])
        tcToTotal = sTCToTotal()
        p = Pool()

        gen.data >> tcToTotal.envelope
        tcToTotal.TCToTotal >> (p, 'lowlevel.tctototal')

        self.assertRaises(RuntimeError, lambda: run(gen))


    def testRegression(self):
        envelope = range(22050)
        envelope.reverse()
        envelope = range(22050) + envelope

        gen = VectorInput(envelope)
        tcToTotal = sTCToTotal()
        p = Pool()

        gen.data >> tcToTotal.envelope
        tcToTotal.TCToTotal >> (p, 'lowlevel.tctototal')

        run(gen)

        self.assertAlmostEqual(p['lowlevel.tctototal'],
                               TCToTotal()(envelope))


suite = allTests(TestTCToTotal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
