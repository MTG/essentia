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
from essentia.streaming import Trimmer

class TestTrimmer_Streaming(TestCase):

    def slice(self, start, end, sr):
        size = 100*sr
        input = range(size)
        startIdx = int(start*sr);
        stopIdx = int(end*sr)
        if stopIdx > size: stopIdx = size
        expected = range(startIdx,stopIdx)
        gen = VectorInput(input)
        pool = Pool()
        trim = Trimmer(startTime = start,
                       endTime = end,
                       sampleRate = sr)

        gen.data >> trim.signal
        trim.signal >> (pool, 'slice')
        run(gen)
        if end != start:
            self.assertEqualVector(pool['slice'], expected)
        else: self.assertEqualVector(pool.descriptorNames(), [])

    def testIntegerSlice(self):
        self.slice(0., 10., 10);
        self.slice(0.2, 25.2, 10);

    def testDecimalSlice(self):
        self.slice(0., 10.43, 10);
        self.slice(0.21, 25.25, 10);
        self.slice(5.13, 10.64, 10);

    def testZeroSizeSlice(self):
        self.slice(5., 5., 10);

    def testTooLargeEndTime(self):
        self.slice(5., 100., 10);
        self.slice(5., 200., 10);

    def testInvalidParams(self):
        self.assertConfigureFails(Trimmer(), {'sampleRate' : 0})
        self.assertConfigureFails(Trimmer(), {'startTime' : -1.0})
        self.assertConfigureFails(Trimmer(), {'endTime' : -1.0})
        self.assertConfigureFails(Trimmer(), {'startTime' : 1.0,
                                              'endTime' : 0})

    def testEmpty(self):
        gen = VectorInput([])
        pool = Pool()
        trim = Trimmer(startTime = 0,
                       endTime = 1,
                       sampleRate = 44100)

        gen.data >> trim.signal
        trim.signal >> (pool, 'slice')
        run(gen)
        self.assertEqualVector(pool.descriptorNames(), [])


suite = allTests(TestTrimmer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
