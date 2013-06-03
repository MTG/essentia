#!/usr/bin/env python

from essentia_test import *
from essentia import *


class TestHFC(TestCase):

    def testRegression(self):
        # Simple regression test, comparing to reference values
        input = readVector(join(filedir(), 'highfrequencycontent/input.txt'))
        expected = readVector(join(filedir(), 'highfrequencycontent/output.txt'))
        self.assertAlmostEqual(HFC(sampleRate=44100, type='Masri')(input), expected[0], 1e-6)
        self.assertAlmostEqual(HFC(sampleRate=44100, type='Jensen')(input), expected[1], 1e-6)
        self.assertAlmostEqual(HFC(sampleRate=44100, type='Brossier')(input), expected[2], 1e-6)


    def testZero(self):
        # Inputting zeros should return zero hfc
        hfc = HFC(type='Masri')(zeros(1024))
        self.assertEqual(hfc, 0)

        hfc = HFC(type='Jensen')(zeros(1024))
        self.assertEqual(hfc, 0)

        hfc = HFC(type='Brossier')(zeros(1024))
        self.assertEqual(hfc, 0)

    def testInvalidParam(self):
        # Test different type than masri, jensen or brossier
        self.assertConfigureFails(HFC(), { 'type':'unknown'})

    def testEmpty(self):
        # Test that empty spectrum yields an exception and spectrum of size 1 returns
        # 0 hfc
        self.assertComputeFails(HFC(), [])

    def testOne(self):
        self.assertEqual(HFC()([9]), 0)


suite = allTests(TestHFC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
