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
from essentia.streaming import RhythmExtractor
from essentia.standard import MonoLoader, RhythmExtractor as stdRhythmExtractor
from math import fabs

class TestRhythmExtractor(TestCase):

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        rhythm = stdRhythmExtractor()

        bpm, _, _, _ = rhythm(audio)
        self.assertAlmostEqualFixedPrecision(bpm, 126, 0) # exact value= 125.726791382


    def runInstance(self, input, tempoHints=None, useOnset=True, useBands=True, poolInit=False):
        print('TestRhythmExtractor: Warning - these tests are evaluated with high tolerances for error, please review these tests')

        gen = VectorInput(input)
        if not tempoHints:
            rhythm = RhythmExtractor(useOnset=useOnset, useBands=useBands)
        else:
            rhythm = RhythmExtractor(useOnset=useOnset, useBands=useBands, tempoHints=array(tempoHints))

        p = Pool()

        gen.data >> rhythm.signal
        rhythm.bpm          >> (p, 'rhythm.bpm')
        rhythm.ticks        >> (p, 'rhythm.ticks')
        rhythm.estimates    >> (p, 'rhythm.estimates')
        rhythm.bpmIntervals >> (p, 'rhythm.bpmIntervals')

        run(gen)

        outputs = ['rhythm.ticks', 'rhythm.estimates',
                   'rhythm.bpmIntervals']

        # in case ther was no output from rhythm extractor in any of the output
        # ports:
        for output in outputs:
            if output not in p.descriptorNames():
                p.add(output, [0])
        if 'rhythm.bpm' not in p.descriptorNames():
            p.set('rhythm.bpm', 0)

        return [ p['rhythm.bpm'],
                 p['rhythm.ticks'],
                 p['rhythm.estimates'],
                 p['rhythm.bpmIntervals'] ]

    def pulseTrain(self, bpm, sr, offset, dur):
        from math import floor
        period = int(floor(sr/(bpm/60.)))
        size = int(floor(sr*dur))
        phase = int(floor(offset*sr))

        if phase > period:
            phase = 0

        impulse = [0.0]*size
        for i in range(size):
            if i%period == phase:
                impulse[i] = 1.0
        return impulse


    def assertVectorWithinVector(self, found, expected, precision=1e-7):
        for i in range(len(found)):
            for j in range(1,len(expected)):
                if found[i] <= expected[j] and found[i] >= expected[j-1]:
                    if fabs(found[i] - expected[j-1]) < fabs(expected[j] - found[i]):
                        self.assertAlmostEqual(found[i], expected[j-1], precision)
                    else:
                        self.assertAlmostEqual(found[i], expected[j], precision)

    def _assertEqualResults(self, result, expected):
        self.assertEqual(result[0], expected[0]) #bpm
        self.assertEqualVector(result[1], expected[1]) # ticks
        self.assertEqualVector(result[2], expected[2]) # estimates
        self.assertEqualVector(result[3], expected[3]) # bpmIntervals


    def testEmpty(self):
        input = []
        expected = [0, 0, 0, 0]
        result = self.runInstance(input, poolInit=True)
        self.assertEqualVector(result, expected)

    def testEmptyUseBands(self):
        input = []
        expected = [0, 0, 0, 0]
        result = self.runInstance(input, useBands=True, useOnset=False, poolInit=True)
        self.assertEqualVector(result, expected)

    def testEmptyUseOnset(self):
        input = []
        expected = [0, 0, 0, 0]
        result = self.runInstance(input, useBands=False, useOnset=True, poolInit=True)
        self.assertEqualVector(result, expected)

    def testZero(self):
        input = [0.0]*100*1024 # 100 frames of size 1024
        expected = [0, [], [], []]
        result = self.runInstance(input, poolInit=True)
        self._assertEqualResults(result, expected)

    def testZeroUseBands(self):
        input = array([0.0]*100*1024) # 100 frames of size 1024
        expected = [0, [], [], []]
        result = self.runInstance(input, useBands=True, useOnset=False, poolInit=True)
        self._assertEqualResults(result, expected)

    def testZeroUseOnset(self):
        input = [0.0]*100*1024 # 100 frames of size 1024
        expected = [0, [], [], []]
        result = self.runInstance(input, useBands=True, useOnset=False, poolInit=True)
        self._assertEqualResults(result, expected)

    def testUseBands(self):
        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)

        # expected values
        expectedTicks = [i/44100. for i in range(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        expectedBpm = 140.
        result = self.runInstance(impulseTrain140, useBands=True, useOnset=False)

        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # estimated bpm
        for i in range(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 1.)

        # bpm intervals
        for i in range(len(result[3])):
            self.assertAlmostEqual(result[3][i], 60./expectedBpm, 0.2)


    def testUseOnset(self):
        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)

        # expected values
        expectedTicks = [i/44100. for i in range(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        expectedBpm = 140.
        result = self.runInstance(impulseTrain140, useBands=False, useOnset=True)

        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # estimated bpm
        for i in range(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 1.)

        # bpm intervals
        for i in range(len(result[3])):
            self.assertAlmostEqual(result[3][i], 60./expectedBpm, 0.2)


    def testImpulseTrain(self):

        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)

        # expected values
        expectedTicks = [i/44100. for i in range(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        expectedBpm = 140.

        result = self.runInstance(impulseTrain140)

        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # estimated bpm
        for i in range(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 1.)

        # bpm intervals
        for i in range(len(result[3])):
            self.assertAlmostEqual(result[3][i], 60./expectedBpm, 0.2)

        # impulse train at 90bpm no offset
        impulseTrain90 = self.pulseTrain(bpm=90., sr=44100., offset=0., dur=20.)

        # impulse train at 200bpm with offset
        impulseTrain200 = self.pulseTrain(bpm=200., sr=44100., offset=.2, dur=10.)

        # impulse train at 200bpm with no offset
        impulseTrain200b = self.pulseTrain(bpm=200., sr=44100., offset=0., dur=25.)

        ## make a new impulse train at 140-90-200-200 bpm
        impulseTrain = impulseTrain140 + impulseTrain90 + impulseTrain200 + \
                       impulseTrain200b

        # expected values

        # ticks
        expectedTicks = [i/44100. for i in range(len(impulseTrain)) if impulseTrain[i]!= 0]

        result = self.runInstance(impulseTrain, expectedTicks)

        # bpm
        expectedBpm = 200
        self.assertAlmostEqual(result[0], expectedBpm, .5)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, 0.03)

        # bpm estimates
        for i in range(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 0.5)

        # bpm intervals: we may need to take into account also multiples of 90,
        # 140 and 200.
        expectedBpmIntervals = [60/90., 60/140., 60/200.]
        self.assertVectorWithinVector(result[3], expectedBpmIntervals)

        ### run w/o tempoHints ###

        result = self.runInstance(impulseTrain)

        expectedBpmVector = [50, 100, 200]

        # bpm: here rhythmextractor is choosing 0.5*expected_bpm, that's why we are
        # comparing the resulting bpm with the expected_bpm_vector:
        self.assertVectorWithinVector([result[0]], expectedBpmVector, 1.)

        # bpm estimates
        self.assertVectorWithinVector(result[2], expectedBpmVector, 0.5)
        self.assertVectorWithinVector(result[1], expectedTicks, 0.03)
        self.assertVectorWithinVector(result[3], expectedBpmIntervals, 0.5)


suite = allTests(TestRhythmExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
