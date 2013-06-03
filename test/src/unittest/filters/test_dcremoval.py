#!/usr/bin/env python
#
# Copyright (C) 2006-2008 Music Technology Group (MTG)
#                         Universitat Pompeu Fabra
#
#

from essentia_test import *

class TestDCRemoval(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(DCRemoval(), { 'sampleRate': -23 })
        self.assertConfigureFails(DCRemoval(), { 'cutoffFrequency': 0 })


    def testEmpty(self):
        self.assertEqualVector(DCRemoval()([]), [])


    def testOneByOne(self):
        # we compare here that filtering an array all at once or the samples
        # one by one will yield the same result
        filt = DCRemoval()
        signal = readVector(join(filedir(), 'filters/x.txt'))

        expected = filt(signal)

        # need to reset the filter here!!
        filt.reset()

        result = []
        for sample in signal:
            result += list(filt([sample]))

        self.assertAlmostEqualVector(result, expected)

    def testZero(self):
        self.assertEqualVector(DCRemoval()(zeros(20)), zeros(20))


    def testConstantInput(self):
        # we only test starting from the 1000th position, because we need to wait
        # for the filter to stabilize
        self.assertAlmostEqualVector(DCRemoval()(ones(20000))[1000:], zeros(20000)[1000:], 3.3e-3)

    def testRegression(self):
        signal = MonoLoader(filename = join(testdata.audio_dir,
                                            'generated', 'doublesize',
                                            'sin_30_seconds.wav'),
                            sampleRate = 44100)()

        dcOffset = 0.2
        dcsignal = signal + dcOffset

        self.assertAlmostEqual(numpy.mean(DCRemoval()(dcsignal)[2500:]),0, 1e-6)


suite = allTests(TestDCRemoval)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
