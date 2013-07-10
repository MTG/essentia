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
from essentia.streaming import Slicer
from numpy import sort

class TestSlicer_Streaming(TestCase):

    def slice(self, startTimes, endTimes):
        nSlices = len(startTimes)
        if nSlices != len(endTimes):
            print "Test cannot be computed"
            exit(1)
        input = range(max(endTimes))

        # expected values:
        expected = []
        orderedTimes = []
        for i in range(nSlices):
            time = (startTimes[i], endTimes[i])
            orderedTimes.append(time)
        orderedTimes = sorted(orderedTimes, lambda x,y:x[0]-y[0])

        for i in range(nSlices):
            expected.append(input[orderedTimes[i][0]:orderedTimes[i][1]])

        gen = VectorInput(input)
        slicer = Slicer(startTimes = startTimes,
                        endTimes = endTimes,
                        timeUnits="samples")
        pool = Pool()
        gen.data >> slicer.audio
        slicer.frame >> (pool, "data")
        run(gen)
        result = pool['data']

        self.assertEqual(len(result), len(expected))
        for i in range(nSlices):
            self.assertEqualVector(result[i], expected[i])

    def testEqualSize(self):
        startTimes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        endTimes =   [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.slice(startTimes, endTimes)

    def testDifferentSize(self):
        startTimes = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
        endTimes =   [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.slice(startTimes, endTimes)

    def testOverlap(self):
        startTimes = [0, 11, 22, 33, 44, 0, 6, 5, 88, 19]
        endTimes   = [30, 60, 45, 100, 50, 60, 10, 50, 100, 99]
        self.slice(startTimes, endTimes)

    def testInvalidParam(self):
        # startTime later than endTime:
        startTimes = [35, 11]
        endTimes   = [30, 60]
        self.assertConfigureFails(Slicer(), {'startTimes' : startTimes,
                                             'endTimes' : endTimes})

        self.assertConfigureFails(Slicer(), {'timeUnits' : 'unknown'})

    def testEmpty(self):
        startTimes = [0, 11]
        endTimes   = [30, 60]
        gen = VectorInput([])
        slicer = Slicer(startTimes = startTimes,
                        endTimes = endTimes,
                        timeUnits="samples")
        pool = Pool()
        gen.data >> slicer.audio
        slicer.frame >> (pool, "data")
        run(gen)
        self.assertEqualVector(pool.descriptorNames(), [])

    def testOneSample(self):
        startTimes = [0]
        endTimes   = [1.0/44100.0]
        gen = VectorInput([1])
        slicer = Slicer(startTimes = startTimes,
                        endTimes = endTimes,
                        timeUnits="seconds")
        pool = Pool()
        gen.data >> slicer.audio
        slicer.frame >> (pool, "data")
        run(gen)
        self.assertEqualVector(pool['data'], [1])

    def testVeryLargeStartAndEndTimes(self):
        # no slices if times are beyond the input length:
        startTimes = [100]
        endTimes   = [101]
        gen = VectorInput([1]*50)
        slicer = Slicer(startTimes = startTimes,
                        endTimes = endTimes,
                        timeUnits="samples")
        pool = Pool()
        gen.data >> slicer.audio
        slicer.frame >> (pool, "data")
        run(gen)
        self.assertEqual(pool.descriptorNames(), [])

    def testEndTimePastEof(self):
        # no slices if times are beyond the input length:
        startTimes = [0]
        endTimes   = [100]
        gen = VectorInput([1])
        slicer = Slicer(startTimes = startTimes,
                        endTimes = endTimes,
                        timeUnits="seconds")
        pool = Pool()
        gen.data >> slicer.audio
        slicer.frame >> (pool, "data")
        run(gen)
        self.assertEqualVector(pool.descriptorNames(), [])

    def Overflow(self):
        self.assertConfigureFails(Slicer(), {'sampleRate' : 44100,
                                             'startTimes' : [2147483649.0],
                                             'endTimes' : [2147483649.5],
                                             'timeUnits' : 'seconds'})

suite = allTests(TestSlicer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
