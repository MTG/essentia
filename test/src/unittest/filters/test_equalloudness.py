#!/usr/bin/env python

from essentia_test import *

class TestEqualLoudness(TestCase):


    def testInvalidParam(self):
        self.assertConfigureFails(EqualLoudness(), { 'sampleRate': -23 })


    def testEmpty(self):
        self.assertEqualVector(EqualLoudness()([]), [])


    def testOneByOne(self):
        # we compare here that filtering an array all at once or the samples
        # one by one will yield the same result
        filt = EqualLoudness()
        signal = readVector(join(filedir(), 'filters/x.txt'))

        expected = filt(signal)

        # need to reset the filter here!!
        filt.reset()

        result = []
        for sample in signal:
            result += list(filt([sample]))

        self.assertAlmostEqualVector(result, expected, 1e-3)

    def testZero(self):
        self.assertEqualVector(EqualLoudness()(zeros(20)), zeros(20))

    def testRegression(self):
        signal = MonoLoader(filename = join(testdata.audio_dir, 'generated', 'doublesize', 'sin_30_seconds.wav'),
                            sampleRate = 44100)()[:100000]
        expected = MonoLoader(filename = join(testdata.audio_dir, 'generated', 'doublesize', 'sin_30_seconds_eqloud.wav'),
                              sampleRate = 44100)()[:100000]

        # assert on the difference of the signals here, because we want the absolute
        # difference, not a relative one
        self.assertAlmostEqualVector(EqualLoudness()(signal) - expected, zeros(len(expected)), 1e-4)


suite = allTests(TestEqualLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
