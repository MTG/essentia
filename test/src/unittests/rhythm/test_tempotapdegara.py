#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
from essentia.standard import TempoTapDegara


class TestTempoTapDegara(TestCase):
    def testDeterministicGap(self):
        # Regular onsets with a missing middle section leave competing
        # Viterbi paths through the gap. The previous process-global rand()
        # noise produced different ticks across repeated calls on this input.
        onsets = zeros(2048)
        onsets[::43] = 1
        onsets[len(onsets)//3:2*len(onsets)//3] = 0

        expected = TempoTapDegara()(onsets)
        tracker = TempoTapDegara()
        for _ in range(5):
            self.assertEqualVector(TempoTapDegara()(onsets), expected)
            self.assertEqualVector(tracker(onsets), expected)


suite = allTests(TestTempoTapDegara)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
