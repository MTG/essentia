#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import RealAccumulator

class TestRealAccumulator_Streaming(TestCase):

    def accumulate(self, input, size = 1):
        # NB: size is not used anymore as RealAccumulator got smarter :-)
        accu = RealAccumulator()
        pool = Pool()
        gen = VectorInput(input)
        gen.data >> accu.data
        accu.array >> (pool, 'accu')
        run(gen)
        if not pool.descriptorNames() : return []
        return pool['accu']

    def testEmpty(self):
        self.assertEqualVector(self.accumulate([]), [])

    def testSingle(self):
        self.assertEqual(self.accumulate([1.0]), [1.0])

    def testRegression(self):
        input = [5.0, 0.0, -1.0, 2.0, -3.0, 4.0];
        result = self.accumulate(input)
        self.assertEqualVector(result, input)

    def testPreferredSize(self):
        # NB: this is not as useful as before when we had a preferredSize parameter
        input = [ float(i) for i in xrange(44100)]
        result = self.accumulate(input, size=1024)
        self.assertEqualVector(result, input)

suite = allTests(TestRealAccumulator_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
