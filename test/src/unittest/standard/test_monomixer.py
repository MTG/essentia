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
from essentia import *
from numpy import sum

class TestMonoMixer_Streaming(TestCase):
    left = []
    right = []

    def clickTrack(self):
        size = 100
        offset = 10
        self.left = [0]*size
        self.right = [0]*size
        for i in range(offset/2, size, offset):
            self.left[i] = 1.0
        for i in range(offset, size, offset):
            self.right[i] = 1

        output = []
        for i in range(size):
            output.append((self.left[i], self.right[i]))

        return array(output)

    def testLeft(self):
        self.assertEqualVector(MonoMixer(type='left')(self.clickTrack(), 2), self.left)

    def testRight(self):
        self.assertEqualVector(MonoMixer(type='right')(self.clickTrack(), 2), self.right)

    def testMix(self):
        mix = MonoMixer(type='mix')(self.clickTrack(), 2)
        self.assertEqual(sum(mix), 19*0.5)

    def testSingle(self):
        mix = MonoMixer(type='mix')(array([(0.9, 0.5)]), 2)
        self.assertAlmostEqual(sum(mix), (0.9+0.5)*0.5)

    def testEmpty(self):
        inputFilename = join(testdata.audio_dir, 'generated', 'empty', 'empty.wav')
        loader = AudioLoader(filename=inputFilename)()
        self.assertEqualVector(MonoMixer(type='left')(loader[0], loader[2]), [])

    def testInvalidParam(self):
        self.assertConfigureFails(MonoMixer(), {'type':'unknown'})


suite = allTests(TestMonoMixer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
