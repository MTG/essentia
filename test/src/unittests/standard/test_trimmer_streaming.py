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
from essentia.streaming import Trimmer
import essentia.standard

class TestTrimmer_Streaming(TestCase):

    def slice(self, start, end, sr):
        size = 100*sr
        input = list(range(size))
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

    # --- sample-precision regression tests (upstream issue 1529 on MTG/essentia) ---
    #
    # Trimmer used to compute startTime*sampleRate in Real (float32). Past
    # 2^24 samples (~380.4 s at 44.1 kHz) the float32 grid spacing exceeds one
    # sample, so cut points drifted: at 3090 s the float32 product is
    # 136268992 instead of 136269000 (-8 samples). The conversion must be done
    # in double and rounded once to the nearest sample.
    #
    # The test signal is a ramp where sample i has value i % MOD, so the value
    # of the first/last output sample encodes its absolute position exactly
    # (MOD is far below 2^24, so every value is exact in float32, unlike a
    # plain ramp whose values would themselves collapse onto the float32 grid
    # at these positions and mask the error being tested).

    MOD = 1 << 20

    def _rampSignal(self, size):
        # built in bounded chunks to avoid a second full-size int64 array
        sig = numpy.empty(size, dtype=numpy.float32)
        chunk = 1 << 22
        for offset in range(0, size, chunk):
            n = min(chunk, size - offset)
            idx = numpy.arange(offset, offset + n, dtype=numpy.int64)
            sig[offset:offset+n] = (idx % self.MOD).astype(numpy.float32)
        return sig

    def _assertExactSlice(self, found, startSample, numSamples):
        found = numpy.asarray(found)
        self.assertEqual(len(found), numSamples)
        expectedFirst = startSample % self.MOD
        expectedLast = (startSample + numSamples - 1) % self.MOD
        self.assertEqual(found[0], expectedFirst)
        self.assertEqual(found[-1], expectedLast)
        idx = numpy.arange(startSample, startSample + numSamples, dtype=numpy.int64)
        expected = (idx % self.MOD).astype(numpy.float32)
        self.assertTrue(numpy.array_equal(found, expected))

    def testLongOffsetSamplePrecisionStreaming(self):
        # trimming a 30 s window at 3090 s must start at exactly sample
        # 136269000 and produce exactly 1323000 samples
        sr = 44100
        startSample = 3090 * sr    # 136269000
        numSamples = 30 * sr       # 1323000
        signal = self._rampSignal(startSample + numSamples + sr)

        gen = VectorInput(signal)
        pool = Pool()
        trim = Trimmer(startTime=3090.0, endTime=3120.0, sampleRate=sr)
        gen.data >> trim.signal
        trim.signal >> (pool, 'slice')
        run(gen)

        self._assertExactSlice(pool['slice'], startSample, numSamples)

    def testLongOffsetSamplePrecisionStandard(self):
        sr = 44100
        startSample = 3090 * sr    # 136269000
        numSamples = 30 * sr       # 1323000
        signal = self._rampSignal(startSample + numSamples + sr)

        trim = essentia.standard.Trimmer(startTime=3090.0, endTime=3120.0,
                                         sampleRate=sr)
        self._assertExactSlice(trim(signal), startSample, numSamples)

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
