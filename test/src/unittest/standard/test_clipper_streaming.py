#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import Clipper
from math import sin, pi

class TestClipper_Streaming(TestCase):

    def testSimple(self):
        sr = 44100
        max = 0.5
        min = -1

        input = [2*sin(2*pi*i*5/sr) for i in xrange(sr)]
        gen = VectorInput(input)
        clip = Clipper(min=min, max=max)
        p = Pool()

        gen.data >> clip.signal >> (p, 'clipped')
        run(gen)

        for val in p['clipped']:
            self.assertTrue(val <= max)
            self.assertTrue(val >= min)

    def testUnClipped(self):
        input = [5, 0, -1, 2, -3, 4]
        expected = input

        gen = VectorInput(input)
        clip = Clipper(min=min(input), max=max(input))
        p = Pool()

        gen.data >> clip.signal >> (p, 'clipped')
        run(gen)

        self.assertEqualVector(p['clipped'], expected)

    def testEmpty(self):
        gen = VectorInput([])
        clip = Clipper()
        p = Pool()

        gen.data >> clip.signal >> (p, 'clipped')
        run(gen)

        self.assertEqualVector(p.descriptorNames(), [])

    def testSingle(self):
        gen = VectorInput([1])
        clip = Clipper()
        p = Pool()

        gen.data >> clip.signal >> (p, 'clipped')
        run(gen)

        self.assertEqualVector(p['clipped'], [1])

    def testStandard(self):
        from essentia.standard import Clipper as stdClipper
        max = 0.5
        min = -1.0
        sr = 44100
        input = [2*sin(2*pi*i*5/sr) for i in xrange(sr)]
        output = stdClipper(max=max, min=min)(input)
        for val in output:
            self.assertTrue(val <= max)
            self.assertTrue(val >= min)



suite = allTests(TestClipper_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
