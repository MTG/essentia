#!/usr/bin/env python

from essentia_test import *
import essentia.streaming as es
from random import randint


class TestDuration(TestCase):

    def testEmpty(self):
        input = []
        self.assertEqual(Duration()(input), 0.0)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(Duration()(input), 0.00226757372729)

    def testOne(self):
        input = [0]
        self.assertAlmostEqual(Duration()(input), 2.26757365454e-5)

        input = [100]
        self.assertAlmostEqual(Duration()(input), 2.26757365454e-5)

    def test30Sec(self):
        input = [randint(0, 100) for x in xrange(44100*30)]
        self.assertAlmostEqual(Duration()(input), 30.0)

    def testSampleRates(self):
        self.assertAlmostEqual(Duration(sampleRate=48000)(zeros(100)), 100./48000.)
        self.assertAlmostEqual(Duration(sampleRate=22050)(zeros(100)), 100./22050.)

    def testBadSampleRate(self):
        self.assertConfigureFails(Duration(), { 'sampleRate' : 0 })

    def testFrameStreaming(self):
        gen = VectorInput([0]*100)
        dur = es.Duration()
        pool = Pool()

        gen.data >> dur.signal
        dur.duration >> (pool, 'duration')
        run(gen)

        self.assertAlmostEqual(pool['duration'], 0.00226757372729)

    def testOneStreaming(self):
        gen = VectorInput([ 23 ])
        dur = es.Duration()
        pool = Pool()

        gen.data >> dur.signal
        dur.duration >> (pool, 'duration')
        run(gen)

        self.assertAlmostEqual(pool['duration'], 2.26757365454e-5)

    def test30SecStreaming(self):
        gen = VectorInput([ randint(0, 100) for x in xrange(44100*30) ])
        dur = es.Duration()
        pool = Pool()

        gen.data >> dur.signal
        dur.duration >> (pool, 'duration')
        run(gen)

        self.assertAlmostEqual(pool['duration'], 30.0)

suite = allTests(TestDuration)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
