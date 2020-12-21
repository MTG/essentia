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
from essentia.standard import MonoLoader, RhythmExtractor2013
from math import fabs

class TestRhythmExtractor2013(TestCase):

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        rhythm = RhythmExtractor2013()

        bpm, _, _, _ ,_= rhythm(audio)
        self.assertAlmostEqualFixedPrecision(bpm, 124.29265, 0) # exact value= 125.726791382

    def runInstance(self, input, method="degara"):
        print('TestRhythmExtractor2013: Warning - these tests are evaluated with high tolerances for error, please review these tests')

        gen = VectorInput(input)
        
        # TODO differentiate Degara and Multinstance scenarios
        rhythm = RhythmExtractor2013(method=method)
        
        #output all the outputs in the natural order in the std mode
        return(rhythm(input))
            rhythm.confidence >> (p, 'rhythm.confidence')

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
        self.assertEqual(result[2], expected[2]) # confidence
        self.assertEqualVector(result[3], expected[3]) # estimates
        self.assertEqualVector(result[4], expected[4]) # bpmIntervals

    #TODO The exact expected values for the result oof passing empty parameters is TBD
    # (see testZero)
    def testEmpty(self):
        input = []
        #expected = [0.0, [], 0.0, [], [],[]]# extra frame for confidence
        #result = self.runInstance(input, method="multifeature")
        #self.assertEqualVector(result, expected)
        #result = self.runInstance(input, method="degara")
        #self.assertEqualVector(result, expected)
        #print(result) 

    def testZero(self):
        input = array([0.0]*10*1024) # 100 frames of size 1024
        expected = [0.0, [], 0.0, [], [],[]] # extra frame for confidence
        result = self.runInstance(input, method="multifeature" )
        self._assertEqualResults(result, expected)
        
        expected = [1033.59375, [0.058049887,0.058049887], 0.0, [], [],[]] # extra frame for confidence
        result = self.runInstance(input, method="degara" )
        self._assertEqualResults(result, expected)

    def testImpulseTrain(self):
        
        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)
        # expected values
        expectedTicks = [i/44100. for i in range(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        expectedBpm = 139.6845550537109
        expectedConfidence =3.785917 
        result = self.runInstance(impulseTrain140,method="multifeature")
        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)
        # ticks
        tickTolerance = 1.214071e+01
        self.assertVectorWithinVector(result[1], expectedTicks, tickTolerance)
        #confidence
        self.assertAlmostEqual(result[2], expectedConfidence, 1e-2)
        # estimated bpm
        for i in range(len(result[3])):
            self.assertAlmostEqual(result[3][i], expectedBpm, 1.)

        # bpm intervals
        for i in range(len(result[4])):
            self.assertAlmostEqual(result[4][i], 60./expectedBpm, 0.2)

        # impulse train at 90bpm no offset
        impulseTrain90 = self.pulseTrain(bpm=90., sr=44100., offset=0., dur=20.)
        
        # impulse train at 200bpm with offset
        impulseTrain200 = self.pulseTrain(bpm=200., sr=44100., offset=.2, dur=10.)

        # impulse train at 200bpm with no offset
        impulseTrain200b = self.pulseTrain(bpm=200., sr=44100., offset=0., dur=25.)

        ## make a new impulse train at 140-90-200-200 bpm
        impulseTrain = impulseTrain140 + impulseTrain90 + impulseTrain200 + \
                       impulseTrain200b
       

        result = self.runInstance(impulseTrain,method="degara")

        # expected values:
        # confidence
        expectedConfidence = 0.0

        # ticks
        expectedTicks = [i/44100. for i in range(len(impulseTrain)) if impulseTrain[i]!= 0]
        
        # bpm
        expectedBpm = 200
        self.assertAlmostEqual(result[0], expectedBpm, .5)

        # ticks
        tickTolerance= 3.0
        self.assertVectorWithinVector(result[1], expectedTicks,tickTolerance )

        #confidence
        expectedConfidence = 0.0
        confidenceTolerance= 0.1

        self.assertAlmostEqual(result[2], expectedConfidence, confidenceTolerance)


        # bpm estimates
        # define tolerance
        for i in range(len(result[3])):
            self.assertAlmostEqual(result[3][i], expectedBpm, 5.0)

        # bpm intervals: we may need to take into account also multiples of 90,
        # 140 and 200.
        expectedBpmIntervals = [60/90., 60/140., 60/200.]
        self.assertVectorWithinVector(result[4], expectedBpmIntervals)
        

        result = self.runInstance(impulseTrain)

        expectedBpmVector = [50, 100, 200]

        # bpm: here rhythmextractor is choosing 0.5*expected_bpm, that's why we are
        # comparing the resulting bpm with the expected_bpm_vector:
        self.assertVectorWithinVector([result[0]], expectedBpmVector, 1.)
        
        #TODO decide on tolerance
        #self.assertVectorWithinVector(result[1], expectedTicks, 0.03)

        self.assertEqual(result[2], expectedConfidence, confidenceTolerance)
       
        self.assertVectorWithinVector(result[3], expectedBpmVector, 0.5)
        
        self.assertVectorWithinVector(result[4], expectedBpmIntervals, 0.05)


suite = allTests(TestRhythmExtractor2013)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
