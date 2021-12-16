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
from essentia.standard import MonoLoader, RhythmExtractor2013
from math import fabs
from math import floor


class TestRhythmExtractor2013(TestCase):

    def testRegressionDegara(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        bpm = RhythmExtractor2013(method="degara")(audio)[0]
        self.assertAlmostEqualFixedPrecision(bpm, 124, 0) 
    
    def testRegressionMultifeature(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        bpm = RhythmExtractor2013( method="multifeature")(audio)[0]
        self.assertAlmostEqualFixedPrecision(bpm, 124, 0) 

    def _runInstance(self, input, method="degara"):
        return RhythmExtractor2013(method=method)(input) 

    def _pulseTrain(self, bpm, sr, offset, dur):
        period = floor(sr/(bpm/60.))
        size = floor(sr*dur)
        phase = floor(offset*sr)

        if phase > period:
            phase = 0

        impulse = [1.0 if i % period == phase else 0 for i in range(size)]
        return impulse

    def _assertVectorWithinVector(self, found, expected, precision=1e-7):
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
        self.assertEqual(result[2], expected[2]) # confidence
        self.assertEqualVector(result[3], expected[3]) # estimates
        self.assertEqualVector(result[4], expected[4]) # bpmIntervals

    def testEmptyMultiFeature(self):
        input = []                          
        self.assertRaises(RuntimeError, lambda: RhythmExtractor2013(method="multifeature")(input))

    def testEmptyDegara(self):
        input = []                          
        self.assertRaises(RuntimeError, lambda: RhythmExtractor2013(method="degara")(input))

    def testZeroMultiFeature(self):
        # TODO: Test currently failing.
        # Non zero BPM values are returned for zero input.
        input = array([0.0]*100*1024) # 100 frames of size 1024
        expected = [0.0, [], 0.0, [], []] 
        result = self._runInstance(input, method="multifeature")
        self._assertEqualResults(result, expected)

    def testZeroDegara(self):
        # TODO: Test currently failing .
        # No zero out of range (738) BPM values are returned for zero input.
        input = array([0.0]*100*1024) # 100 frames of size 1024
        expected = [0.0, [], 0.0, [], []] 
        result = self._runInstance(input, method="degara")
        self._assertEqualResults(result, expected)

    def _createPulseTrain140(self): 
        return self._pulseTrain(bpm=140, sr=44100., offset=.1, dur=10) 

    def _createPulseTrainCombo(self):
        impulseTrain90 = self._pulseTrain(bpm=90., sr=44100., offset=0., dur=20.)
        impulseTrain140 = self._pulseTrain(bpm=140., sr=44100., offset=.1, dur=10.)
        impulseTrain200 = self._pulseTrain(bpm=200., sr=44100., offset=.2, dur=10.)
        impulseTrain200b = self._pulseTrain(bpm=200., sr=44100., offset=0., dur=25.)
        # Result is a combined impulse train with 90-140-200-bpm and different offsets.
        return impulseTrain90 + impulseTrain140 + impulseTrain200 + impulseTrain200b 
    
    def testImpulseTrain140Multifeature(self):
        impulseTrain140=self._createPulseTrain140()
        bpm, ticks, confidence, estimates, bpmIntervals = self._runInstance(impulseTrain140,method="multifeature")
        
        expectedBpm = 140
        expectedTicks = [i/44100. for i in range(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        
        # In this tests a 200 ms tolerance for ticks is a good enough compromise 
        # Hence the value 0.2 for BPM tolerance.
        # In other algos (e.g. tempotapmaxagreement) the max. confidence values is 5.32
        # Ballpark figure of 3.8 were observed in tests here.
        # This is approx. 70% of max conf. value in tempotapmaxagreement
        # On top of this we allow a margin of approx 20%
        # Hence, the tolerance for expectedConfidence is  1.0.

        expectedConfidence =3.8 
        self.assertAlmostEqual(bpm, expectedBpm, 0.2)
        self._assertVectorWithinVector(ticks, expectedTicks, 0.2)
        self.assertAlmostEqual(confidence, expectedConfidence, 1.0)

        for i in range(len(estimates)):
            self.assertAlmostEqual(estimates[i], expectedBpm, .5)

        for i in range(len(bpmIntervals)):
            self.assertAlmostEqual(bpmIntervals[i], 60./expectedBpm, 0.2)

    def testImpulseTrainComboMultifeature(self):
        bpm, ticks, confidence, estimates, bpmIntervals = self._runInstance(self._createPulseTrainCombo() , method="multifeature")
        
        expectedBpm = 117 
        expectedConfidence = 3.0 # This value was found by doing sample run tests and logging output.

        # It is unrealistic to achieve a good match in ticks for the particular combo
        # Therefore we omit assert for tick positions and bpm intervals
        self.assertAlmostEqual(bpm, expectedBpm, .5)
        self.assertAlmostEqual(confidence, expectedConfidence, 1.0)
        
        # We widen the margin on expected BPM differences since we are dealing with combined BPMs.
        for i in range(len(estimates)):
            self.assertAlmostEqual(estimates[i], expectedBpm, 5.0)

    def testImpulseTrain140Degara(self):
        impulseTrain140=self._createPulseTrain140()
        bpm, ticks, confidence, estimates, bpmIntervals = self._runInstance(impulseTrain140, method="degara")
        
        expectedBpm = 140
        expectedTicks = [i/44100. for i in range(len(impulseTrain140)) if impulseTrain140[i]!= 0]  
        expectedConfidence  = 0.0  # For degara thee is no confidence, we expect this value to be zero.
  
        self.assertAlmostEqual(bpm, expectedBpm, 0.2)
        self._assertVectorWithinVector(ticks, expectedTicks, 0.2)
        self.assertAlmostEqual(confidence, expectedConfidence)
    
        for i in range(len(estimates)):
            self.assertAlmostEqual(estimates[i], expectedBpm, 0.2)
        for i in range(len(bpmIntervals)):
            self.assertAlmostEqual(bpmIntervals[i], 60./expectedBpm, 0.2)
          
    def testImpulseTrainComboDegara(self):
        bpm, ticks, confidence, estimates, bpmIntervals = self._runInstance(self._createPulseTrainCombo(), method="degara")
      
        expectedBpm = 100          # Hard to determine. It would be a hybrid of 90,140 and 200 bpm
        expectedConfidence  = 0.0  # For degara thee is no confidence, we expect this value to be zero.

        # It is unrealistic to achieve a good match in ticks for the particular combo.
        # Therefore we omit assert for tick positions and bpm intervals
        self.assertAlmostEqual(bpm, expectedBpm, .5)
        self.assertAlmostEqual(confidence, expectedConfidence)

        # We widen the margin on expected BPM differences since we are dealing with combined BPMs.
        for i in range(len(estimates)):
            self.assertAlmostEqual(estimates[i], expectedBpm, 5.0)


suite = allTests(TestRhythmExtractor2013)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
