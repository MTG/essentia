#!/usr/bin/env python

from essentia_test import *


class TestDynamicComplexity(TestCase):

    def testEmpty(self):
        self.assertEqualVector(DynamicComplexity()([]), (0, -90))

    def testOne(self):
        self.assertEqualVector(DynamicComplexity()([10]), (0, -90))

    def testSilence(self):
        self.assertEqualVector(DynamicComplexity()([0]*44100), (0, -90))

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded', 'roxette.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        self.assertAlmostEqualVector(DynamicComplexity()(audio),
                                     (1.904192328453064, -17.803413391113281),1e-1)

    def testStreamingRegression(self):
        from essentia.streaming import MonoLoader as sMonoLoader, \
                                       DynamicComplexity as sDynamicComplexity

        filename = join(testdata.audio_dir, 'recorded', 'roxette.wav')
        loader = sMonoLoader(filename=filename, downmix='left', sampleRate=44100)
        dyn = sDynamicComplexity()
        pool = Pool()

        loader.audio >> dyn.signal
        dyn.dynamicComplexity >> (pool, 'complexity')
        dyn.loudness >> (pool, 'loudness')
        run(loader)

        self.assertAlmostEqual(pool['complexity'], 1.904192328453064, 1e-1)
        self.assertAlmostEqual(pool['loudness'], -17.803413391113281, 1e-1)


suite = allTests(TestDynamicComplexity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
