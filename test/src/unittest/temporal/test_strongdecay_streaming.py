#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import StrongDecay as sStrongDecay

class TestStrongDecay_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        strongDecay = sStrongDecay()
        p = Pool()

        gen.data >> strongDecay.signal
        strongDecay.strongDecay >> (p, 'strongDecay')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testOneValue(self):
        gen = VectorInput([1.0])
        strongDecay = sStrongDecay()
        p = Pool()

        gen.data >> strongDecay.signal
        strongDecay.strongDecay >> (p, 'strongDecay')

        self.assertRaises(EssentiaException, lambda: run(gen))


    def testRegression(self):
        # borrowing lpc's input vector for this regression test
        input = readVector(join(filedir(), 'lpc', 'input.txt'))

        gen = VectorInput(input)
        strongDecay = sStrongDecay()
        p = Pool()

        gen.data >> strongDecay.signal
        strongDecay.strongDecay >> (p, 'strongDecay')

        run(gen)

        self.assertAlmostEqual(p['strongDecay'], StrongDecay()(input), 1e-6)


suite = allTests(TestStrongDecay_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
