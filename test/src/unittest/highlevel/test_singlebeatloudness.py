#!/usr/bin/env python

from numpy import *
from essentia_test import *

class TestSingleBeatLoudness(TestCase):

    def testZero(self):
        self.assertEqualVector(SingleBeatLoudness(frequencyBands = [20,150])(zeros(8192)), zeros(2))

    def testEmpty(self):
        self.assertComputeFails(SingleBeatLoudness(), [])

    def testSingle(self):
        self.assertComputeFails(SingleBeatLoudness(), [1])

    def testInvalidInput(self):
        loudness = SingleBeatLoudness(sampleRate=44100,
                                      beatWindowDuration=0.1,
                                      beatDuration=0.05)
        #should fail due to input.size < beatDuration+beatWindowDuration
        self.assertComputeFails(SingleBeatLoudness(), [i for i in range(1000)])

    def testInvalidParam(self):
        self.assertConfigureFails(SingleBeatLoudness(), {'beatWindowDuration':0.05,
                                                         'beatDuration':0.1})

        self.assertConfigureFails(SingleBeatLoudness(), {'beatWindowDuration':0.1,
                                                         'beatDuration':-0.05})

        self.assertConfigureFails(SingleBeatLoudness(), {'beatWindowDuration':-0.1,
                                                         'beatDuration':0.05})

        self.assertConfigureFails(SingleBeatLoudness(), {'sampleRate':-441000})

    def testRegression(self):
        # test that it yields valid output (which doesn't mean correct output ;)
        loudness = SingleBeatLoudness(frequencyBands = [20,150])(array(random.rand(8192)))
        self.assert_(not any(numpy.isnan(loudness)))
        self.assert_(not any(numpy.isinf(loudness)))
        self.assert_(all(array(loudness) >= 0.0))


suite = allTests(TestSingleBeatLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
