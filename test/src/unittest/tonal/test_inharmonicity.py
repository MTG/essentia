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


class TestInharmonicity(TestCase):

    def testInvalidInput(self):
        # frequencies should be in ascendent order
        freqs = [440., 100., 660.]
        mags = ones(len(freqs))
        self.assertComputeFails(Inharmonicity(), freqs, mags)
        # frequencies cannot be duplicated
        freqs = [440., 440., 660.]
        self.assertComputeFails(Inharmonicity(), freqs, mags)
        # freqs and mags must have same size
        freqs = [100., 440., 660.]
        mags = ones(len(freqs)-1)
        self.assertComputeFails(Inharmonicity(), freqs, mags)

    def testPurelyHarmonic(self):
        f0 = 110
        freqs = [0.5, 1.0, 2.0, 3.0, 4.0]
        freqs = [freq*f0 for freq in freqs]
        mags = ones(len(freqs))
        self.assertEqual(Inharmonicity()(freqs, mags), 0)

        freqs = [1.0, 5.0, 8.0, 15.0]
        freqs = [freq*f0 for freq in freqs]
        mags = ones(len(freqs))
        self.assertEqual(Inharmonicity()(freqs, mags), 0)


    def testRegression(self):
        # frequencies should be in ascendent order
        f0 = 110
        freqs = [0.5, 0.75, 1.0, 2.0, 3.5, 4, 4.2, 4.9, 6]
        freqs = [freq*f0 for freq in freqs]
        mags = ones(len(freqs))
        self.assertAlmostEqual(Inharmonicity()(freqs, mags),0.122222222388)

        semitones = [0.5, 0.75, 12.5, 14.0, 24.1]
        freqs = [pow(2.0, s/12.0)*f0 for s in semitones]
        mags = ones(len(freqs))
        self.assertAlmostEqual(Inharmonicity()(freqs, mags),0.0573841929436)

        from random import random
        semitones = [(1+random())*i for i in range(10)]
        semitones.sort()
        freqs = [pow(2.0, s/12.0)*f0 for s in semitones]
        mags = ones(len(freqs))
        self.assertTrue(Inharmonicity()(freqs, mags) > 0)

    def testEmpty(self):
        self.assertEqual(Inharmonicity()([],[]), 0)

    def testZero(self):
        freqs = zeros(10)
        mags = ones(len(freqs))
        self.assertComputeFails(Inharmonicity(), freqs, mags)

    def testDC(self):
        freqs = [0, 100, 200]
        mags = ones(len(freqs))
        self.assertComputeFails(Inharmonicity(), freqs, mags)

    def testOnePeak(self):
        freqs = [100]
        mags = ones(len(freqs))
        self.assertEqual(Inharmonicity()(freqs, mags), 0)



suite = allTests(TestInharmonicity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
