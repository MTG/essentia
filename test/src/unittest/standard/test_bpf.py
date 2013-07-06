#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/



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
