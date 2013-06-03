#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import Leq as sLeq
from math import sin, pi

class TestLeq_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        leq = sLeq()
        p = Pool()

        gen.data >> leq.signal
        leq.leq >> (p, 'leq')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testRegression(self):
        input = [1. / (i+1) * sin(2*pi*440*i/44100) for i in range(22050)]
        input += [0]*22050

        gen = VectorInput(input)
        leq = sLeq()
        p = Pool()

        gen.data >> leq.signal
        leq.leq >> (p, 'leq')

        run(gen)

        self.assertAlmostEqual(p['leq'], Leq()(input), 5e-5)


suite = allTests(TestLeq_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
