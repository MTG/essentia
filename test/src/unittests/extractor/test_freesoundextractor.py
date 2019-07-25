#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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


class TestFreesoundExtractor(TestCase):

    def testEmpty(self):
        inputFilename = join(testdata.audio_dir, 'generated', 'empty', 'empty.aiff')
        # NOTE: AudioLoader will through exception on "empty.wav" complaining that
        # it cannot read stream info, using "empty.aiff" therefore...
        self.assertRaises(RuntimeError, lambda: FreesoundExtractor()(inputFilename))

    def testSilence(self):
        inputFilename = join(testdata.audio_dir, 'generated', 'silence', 'silence.flac')
        self.assertRaises(RuntimeError, lambda: FreesoundExtractor()(inputFilename))
        return

    def testCorruptFile(self):
        inputFilename = join(testdata.audio_dir, 'generated', 'unsupported.au')
        self.assertRaises(RuntimeError, lambda: FreesoundExtractor()(inputFilename))

    def testComputeValid(self):
        # Simply checks if computation succeeded. Ideally, we would need
        # a regression test for each descriptor in the pool.
        inputFilename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')
        pool, poolFrames = FreesoundExtractor()(inputFilename)
        self.assertValidPool(pool)
        self.assertValidPool(poolFrames)

    def testRobustness(self):
        # TODO test that computed descriptors are similar across formats
        return

    def testLengthMetadata(self):
        inputFilename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        pool, _ = FreesoundExtractor(startTime=10, endTime=40)(inputFilename)
        self.assertAlmostEqualFixedPrecision(pool['metadata.audio_properties.length'], 45.43, 2)
        self.assertAlmostEqualFixedPrecision(pool['metadata.audio_properties.analysis.length'], 30., 2)


suite = allTests(TestFreesoundExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
