#!/usr/bin/env python

from essentia_test import *

class TestBPF(TestCase):

    def testBadParams(self):
        # Tries to configure BPF with x and y that don't match
        bpf = BPF()

        self.assertConfigureFails(BPF(), { 'xPoints': [], 'yPoints': [] })

        self.assertConfigureFails(BPF(), { 'xPoints': [ 0, 10 ],
                                           'yPoints': [ 0 ] })

        self.assertConfigureFails(BPF(), { 'xPoints': [ 0, 10, 10, 20 ],
                                           'yPoints': [ 0, 5, -23, 17 ] })

    def testSimple(self):
        # Test it works correctly for standard input, also test on edge points
        bpf = BPF(xPoints = [0, 10, 20],
                  yPoints = [0, 20, 0])

        self.assertEqual(bpf(0.), 0.)
        self.assertEqual(bpf(5.), 10.)
        self.assertEqual(bpf(5.5), 11.)
        self.assertEqual(bpf(15.), 10.)
        self.assertEqual(bpf(20.), 0.)
        self.assertEqual(bpf(10.), 20.)

    def testInvalidInput(self):
        # Test that BPF only returns values for the range specified when configuring
        bpf = BPF(xPoints = [0, 10],
                  yPoints = [0, 10])

        self.assertRaises(EssentiaException, bpf, -1)
        self.assertRaises(EssentiaException, bpf, 11)

    def testDecreasingX(self):
        # Test that bpf checks whether the x points are ordered
        self.assertConfigureFails(BPF(), { 'xPoints': [ 10, 0 ],
                                           'yPoints': [ 0, 10 ] })


suite = allTests(TestBPF)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
