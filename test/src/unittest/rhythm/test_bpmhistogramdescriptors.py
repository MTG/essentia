#!/usr/bin/env python

from essentia_test import *

class TestBPMHistogramDescriptors(TestCase):

    def testEmpty(self):
        bpm1, weight1, spread1, bpm2, weight2, spread2 = BPMHistogramDescriptors()([])
        self.assertEqual(bpm1, 0)
        self.assertEqual(weight1, 0)
        self.assertEqual(spread1, 0)
        self.assertEqual(bpm2, 0)
        self.assertEqual(weight2, 0)
        self.assertEqual(spread2, 0)

    def testZero(self):
        bpm1, weight1, spread1, bpm2, weight2, spread2 = BPMHistogramDescriptors()([0])
        self.assertEqual(bpm1, 0)
        self.assertEqual(weight1, 0)
        self.assertEqual(spread1, 0)
        self.assertEqual(bpm2, 0)
        self.assertEqual(weight2, 0)
        self.assertEqual(spread2, 0)

    def testOne(self):
        bpm1, weight1, spread1, bpm2, weight2, spread2 = BPMHistogramDescriptors()([0.5])
        self.assertEqual(bpm1, 120)
        self.assertEqual(weight1, 1)
        self.assertEqual(spread1, 0)
        self.assertEqual(bpm2, 0)
        self.assertEqual(weight2, 0)
        self.assertEqual(spread2, 0)

    def testAbnormalValues(self):
        bpms = [-100, 300]
        intervals = []
        for bpm in bpms:
            intervals.append(60. / bpm)
        bpm1, weight1, spread1, bpm2, weight2, spread2 = BPMHistogramDescriptors()(intervals)
        self.assertEqual(bpm1, 0)
        self.assertEqual(weight1, 0)
        self.assertEqual(spread1, 0)
        self.assertEqual(bpm2, 0)
        self.assertEqual(weight2, 0)
        self.assertEqual(spread2, 0)

    def testRounding(self):
        bpm1, weight1, spread1, bpm2, weight2, spread2 = BPMHistogramDescriptors()([60. / 100.5])
        self.assertEqual(bpm1, 101)

    def testRegression(self):
        bpms = [118, 119, 120, 120, 121, 122, 98, 99, 99, 100, 100, 100, 100, 101, 101, 102]
        intervals = []
        for bpm in bpms:
            intervals.append(60. / bpm)
        intervals.append(0) # Add an extra zero to check if it gets properly dropped
        bpm1, weight1, spread1, bpm2, weight2, spread2 = BPMHistogramDescriptors()(intervals)
        self.assertAlmostEqual(bpm1, 100, 1e-5)
        self.assertAlmostEqual(weight1, 0.5, 1e-5)
        self.assertAlmostEqual(spread1, 0.2, 1e-5)
        self.assertAlmostEqual(bpm2, 120, 1e-5)
        self.assertAlmostEqual(weight2, 0.25, 1e-5)
        self.assertAlmostEqual(spread2, 0.333333, 1e-5)


suite = allTests(TestBPMHistogramDescriptors)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

