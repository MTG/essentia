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
        print 'TestRhythmExtractor: Warning - these tests are evaluated with high tolerances for error, please review these tests'

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
        #rhythm.rubatoStart  >> (p, 'rhythm.rubatoStart')
        #rhythm.rubatoStop   >> (p, 'rhythm.rubatoStop')
        rhythm.bpmIntervals >> (p, 'rhythm.bpmIntervals')

        run(gen)

        outputs = ['rhythm.ticks', 'rhythm.estimates',
                   #'rhythm.rubatoStart', 'rhythm.rubatoStop', 
                   'rhythm.bpmIntervals']

        # in case ther was no output from rhythm extractor in any of the output
        # ports, specially common for rubato start/stop:
        for output in outputs:
            if output not in p.descriptorNames():
                p.add(output, [0])
        if 'rhythm.bpm' not in p.descriptorNames():
            p.set('rhythm.bpm', 0)

        return [ p['rhythm.bpm'],
                 p['rhythm.ticks'],
                 p['rhythm.estimates'],
                 #p['rhythm.rubatoStart'],
                 #p['rhythm.rubatoStop'],
                 p['rhythm.bpmIntervals'] ]

    def pulseTrain(self, bpm, sr, offset, dur):
        from math import floor
        period = int(floor(sr/(bpm/60.)))
        size = int(floor(sr*dur))
        phase = int(floor(offset*sr))

        if phase > period:
            phase = 0

        impulse = [0.0]*size
        for i in xrange(size):
            if i%period == phase:
                impulse[i] = 1.0
        return impulse


    def assertVectorWithinVectorDifference(self, found, expected, precision=1e-7):
        for i in xrange(len(found)):
            for j in xrange(1,len(expected)):
                if found[i] <= expected[j] and found[i] >= expected[j-1]:
                    if fabs(found[i] - expected[j-1]) < fabs(expected[j] - found[i]):
                        self.assertAlmostEqual(found[i]-expected[j-1], 0, precision)
                    else:
                        self.assertAlmostEqual(found[i]-expected[j], 0, precision)


    def assertVectorWithinVector(self, found, expected, precision=1e-7):
        for i in xrange(len(found)):
            for j in xrange(1,len(expected)):
                if found[i] <= expected[j] and found[i] >= expected[j-1]:
                    if fabs(found[i] - expected[j-1]) < fabs(expected[j] - found[i]):
                        self.assertAlmostEqual(found[i], expected[j-1], precision)
                    else:
                        self.assertAlmostEqual(found[i], expected[j], precision)

    def _assertEqualResults(self, result, expected):
        self.assertEqual(result[0], expected[0]) #bpm
        self.assertEqualVector(result[1], expected[1]) # ticks
        self.assertEqualVector(result[2], expected[2]) # estimates
        #self.assertEqualMatrix(result[3], expected[3]) # rubatoStart
        #self.assertEqualMatrix(result[4], expected[4]) # rubatoStop
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
        input = [0.0]*10*1024 # 100 frames of size 1024
        expected = [0, [], [], []]
        result = self.runInstance(input, poolInit=True)
        self._assertEqualResults(result, expected)

    def testZeroUseBands(self):
        input = array([0.0]*10*1024) # 100 frames of size 1024
        expected = [0, [], [], []]
        result = self.runInstance(input, useBands=True, useOnset=False, poolInit=True)
        self._assertEqualResults(result, expected)

    def testZeroUseOnset(self):
        input = [0.0]*10*1024 # 100 frames of size 1024
        expected = [0, [], [], []]
        result = self.runInstance(input, useBands=True, useOnset=False, poolInit=True)
        self._assertEqualResults(result, expected)

    def testUseBands(self):
        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)

        # expected values
        expectedTicks = [i/44100. for i in xrange(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        expectedBpm = 140.
        expectedRubatoStart = []
        expectedRubatoStop = []
        result = self.runInstance(impulseTrain140, useBands=True, useOnset=False)

        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # estimated bpm
        for i in xrange(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 1.)

        # bpm intervals
        for i in xrange(len(result[3])):
            self.assertAlmostEqual(result[3][i], 60./expectedBpm, 0.2)

        ## rubato start/stop
        #self.assertEqualVector(result[3], expectedRubatoStart)
        #self.assertEqualVector(result[4], expectedRubatoStop)


    def testUseOnset(self):
        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)

        #show(plot(impulseTrain140))
        # expected values
        expectedTicks = [i/44100. for i in xrange(len(impulseTrain140)) if impulseTrain140[i]!= 0]
        expectedBpm = 140.
        expectedRubatoStart = []
        expectedRubatoStop = []
        result = self.runInstance(impulseTrain140, useBands=False, useOnset=True)

        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # estimated bpm
        for i in xrange(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 1.)

        # bpm intervals
        for i in xrange(len(result[3])):
            self.assertAlmostEqual(result[3][i], 60./expectedBpm, 0.2)

        ## rubato start/stop
        #self.assertEqualVector(result[3], expectedRubatoStart)
        #self.assertEqualVector(result[4], expectedRubatoStop)


    def testImpulseTrain(self):
        
        impulseTrain140 = self.pulseTrain(bpm=140, sr=44100., offset=.1, dur=10)
        
        # expected values
        expectedTicks = [i/44100. for i in xrange(len(impulseTrain140)) if impulseTrain140[i]!= 0]

        expectedBpm = 140.
        ## Rubato should be empty in this case, however due to the nature of the
        ## test [0] is manually added to the pool:
        #expectedRubatoStart = []
        #expectedRubatoStop = []

        result = self.runInstance(impulseTrain140)

        # bpm
        self.assertAlmostEqual(result[0], expectedBpm, 1e-2)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # estimated bpm
        for i in xrange(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 1.)

        # bpm intervals
        for i in xrange(len(result[3])):
            self.assertAlmostEqual(result[3][i], 60./expectedBpm, 0.2)

        ## rubato start/stop
        #self.assertEqualVector(result[3], expectedRubatoStart)
        #self.assertEqualVector(result[4], expectedRubatoStop)
        
        # impulse train at 90bpm no offset
        impulseTrain90 = self.pulseTrain(bpm=60., sr=44100., offset=0., dur=20.)
        
        # impulse train at 200bpm with offset
        impulseTrain200 = self.pulseTrain(bpm=200., sr=44100., offset=.2, dur=10.)

        # impulse train at 200bpm with no offset
        impulseTrain200b = self.pulseTrain(bpm=200., sr=44100., offset=0., dur=25.)

        ## make a new impulse train at 140-90-200-200 bpm
        impulseTrain = impulseTrain140 + impulseTrain90 + impulseTrain200 + \
                       impulseTrain200b
        
        # expected values

        # ticks
        expectedTicks = [i/44100. for i in xrange(len(impulseTrain)) if impulseTrain[i]!= 0]
        
        result = self.runInstance(impulseTrain, expectedTicks)

        # bpm
        expectedBpm = 200
        self.assertAlmostEqual(result[0], expectedBpm, .5)

        # ticks
        self.assertVectorWithinVector(result[1], expectedTicks, .1)

        # bpm estimates
        for i in xrange(len(result[2])):
            self.assertAlmostEqual(result[2][i], expectedBpm, 0.5)

        # bpm intervals: we may need to take into account also multiples of 90,
        # 140 and 200.
        expectedBpmIntervals = [60/90., 60/140., 60/200.]
        self.assertVectorWithinVector(result[3], expectedBpmIntervals)
        
        ## rubato start/stop
        #expectedRubatoStart = [10]
        #expectedRubatoStop = [30]
        
        #self.assertAlmostEqualVector(result[3], expectedRubatoStart, 0.3)
        #self.assertAlmostEqualVector(result[4], expectedRubatoStop, 0.03)
        
        ### run w/o tempoHints ###

        result = self.runInstance(impulseTrain)

        expectedBpmVector = [50, 100, 200]

        # bpm: here rhythmextractor is choosing 0.5*expected_bpm, that's why we are
        # comparing the resulting bpm with the expected_bpm_vector:
        self.assertVectorWithinVector([result[0]], expectedBpmVector, 1.)
        
        # bpm estimates
        self.assertVectorWithinVector(result[2], expectedBpmVector, 0.5)
        
        # ticks # TODO ticks results fail test
        self.assertVectorWithinVector(result[1], expectedTicks, 0.03)

        # bpm intervals # TODO ticks results fail test
        self.assertVectorWithinVector(result[3], expectedBpmIntervals, 0.5)
        
        ## rubato start/stop
        #self.assertAlmostEqualVector(result[3], expectedRubatoStart, .3)
        #self.assertAlmostEqualVector(result[4], expectedRubatoStop, .03)
   
    """ 
    def testRubato(self):
        # beats extracted from bpmrubato test:
        from numpy import mean
        beats = [0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9, 4.5, 5.1, 5.7, 6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5, 11.1, 11.7, 12.3, 12.9, 13.5, 14.1, 14.7, 15.3, 15.9, 16.5, 17.1, 17.7, 18.3, 18.9, 19.5, 20.1, 20.7, 21.3, 21.9, 22.5, 23.1, 23.7, 24.3, 24.9, 25.5, 26.1, 26.7, 27.3, 27.9, 28.5, 29.1, 29.7, 30.3, 30.9, 31.5, 32.1, 32.7, 33.3, 33.9, 34.5, 35.1, 35.7, 36.3, 36.9, 37.5, 38.1, 38.7, 39.3, 39.9, 40.5, 41.1, 41.7, 42.3, 42.9, 43.5, 44.1, 44.7, 45.3, 45.9, 46.5, 47.1, 47.7, 48.3, 48.9, 49.5, 50.1, 50.7, 51.3, 51.9, 52.5, 53.1, 53.7, 54.3, 54.9, 55.5, 56.1, 56.7, 57.3, 57.9, 58.5, 59.1, 59.7, 60.3, 60.7285714286, 61.1571428571, 61.5857142857, 62.0142857143, 62.4428571429, 62.8714285714, 63.3, 63.7285714286, 64.1571428571, 64.5857142857, 65.0142857143, 65.4428571429, 65.8714285714, 66.3, 66.7285714286, 67.1571428571, 67.5857142857, 68.0142857143, 68.4428571429, 68.8714285714, 69.3, 69.7285714286, 70.1571428571, 70.5857142857, 71.0142857143, 71.4428571429, 71.8714285714, 72.3, 72.7285714286, 73.1571428571, 73.5857142857, 74.0142857143, 74.4428571429, 74.8714285714, 75.3, 75.7285714286, 76.1571428571, 76.5857142857, 77.0142857143, 77.4428571429, 77.8714285714, 78.3, 78.7285714286, 79.1571428571, 79.5857142857, 80.0142857143, 80.4428571429, 80.8714285714, 81.3, 81.7285714286, 82.1571428571, 82.5857142857, 83.0142857143, 83.4428571429, 83.8714285714, 84.3, 84.7285714286, 85.1571428571, 85.5857142857, 86.0142857143, 86.4428571429, 86.8714285714, 87.3, 87.7285714286, 88.1571428571, 88.5857142857, 89.0142857143, 89.4428571429, 89.8714285714, 90.3, 90.7285714286, 91.1571428571, 91.5857142857, 92.0142857143, 92.4428571429, 92.8714285714, 93.3, 93.7285714286, 94.1571428571, 94.5857142857, 95.0142857143, 95.4428571429, 95.8714285714, 96.3, 96.7285714286, 97.1571428571, 97.5857142857, 98.0142857143, 98.4428571429, 98.8714285714, 99.3, 99.7285714286, 100.157142857, 100.585714286, 101.014285714, 101.442857143, 101.871428571, 102.3, 102.728571429, 103.157142857, 103.585714286, 104.014285714, 104.442857143, 104.871428571, 105.3, 105.728571429, 106.157142857, 106.585714286, 107.014285714, 107.442857143, 107.871428571, 108.3, 108.728571429, 109.157142857, 109.585714286, 110.014285714, 110.442857143, 110.871428571, 111.3, 111.728571429, 112.157142857, 112.585714286, 113.014285714, 113.442857143, 113.871428571, 114.3, 114.728571429, 115.157142857, 115.585714286, 116.014285714, 116.442857143, 116.871428571, 117.3, 117.728571429, 118.157142857, 118.585714286, 119.014285714, 119.442857143, 119.871428571, 120.171428571, 120.6, 121.028571429, 121.457142857, 121.885714286, 122.314285714, 122.742857143, 123.171428571, 123.6, 124.028571429, 124.457142857, 124.885714286, 125.314285714, 125.742857143, 126.171428571, 126.6, 127.028571429, 127.457142857, 127.885714286, 128.314285714, 128.742857143, 129.171428571, 129.6, 130.028571429, 130.457142857, 130.885714286, 131.314285714, 131.742857143, 132.171428571, 132.6, 133.028571429, 133.457142857, 133.885714286, 134.314285714, 134.742857143, 135.171428571, 135.6, 136.028571429, 136.457142857, 136.885714286, 137.314285714, 137.742857143, 138.171428571, 138.6, 139.028571429, 139.457142857, 139.885714286, 140.314285714, 140.742857143, 141.171428571, 141.6, 142.028571429, 142.457142857, 142.885714286, 143.314285714, 143.742857143, 144.171428571, 144.6, 145.028571429, 145.457142857, 145.885714286, 146.314285714, 146.742857143, 147.171428571, 147.6, 148.028571429, 148.457142857, 148.885714286, 149.314285714, 149.742857143, 150.171428571, 150.6, 151.028571429, 151.457142857, 151.885714286, 152.314285714, 152.742857143, 153.171428571, 153.6, 154.028571429, 154.457142857, 154.885714286, 155.314285714, 155.742857143, 156.171428571, 156.6, 157.028571429, 157.457142857, 157.885714286, 158.314285714, 158.742857143, 159.171428571, 159.6, 160.028571429, 160.457142857, 160.885714286, 161.314285714, 161.742857143, 162.171428571, 162.6, 163.028571429, 163.457142857, 163.885714286, 164.314285714, 164.742857143, 165.171428571, 165.6, 166.028571429, 166.457142857, 166.885714286, 167.314285714, 167.742857143, 168.171428571, 168.6, 169.028571429, 169.457142857, 169.885714286, 170.314285714, 170.742857143, 171.171428571, 171.6, 172.028571429, 172.457142857, 172.885714286, 173.314285714, 173.742857143, 174.171428571, 174.6, 175.028571429, 175.457142857, 175.885714286, 176.314285714, 176.742857143, 177.171428571, 177.6, 178.028571429, 178.457142857, 178.885714286, 179.314285714, 179.742857143, 180.171428571, 180.471428571, 181.071428571, 181.671428571, 182.271428571, 182.871428571, 183.471428571, 184.071428571, 184.671428571, 185.271428571, 185.871428571, 186.471428571, 187.071428571, 187.671428571, 188.271428571, 188.871428571, 189.471428571, 190.071428571, 190.371428571, 190.911969112, 191.447683398, 191.978656849, 192.504972638, 193.026711769, 193.543953148, 194.056773661, 194.565248237, 195.069449918, 195.569449918, 196.065317687, 196.557120965, 197.044925843, 197.528796811, 198.008796811, 198.484987287, 198.957428232, 199.426178232, 199.891294511, 200.352832973, 200.652832973, 201.203291688, 201.758847244, 202.319594907, 202.885632643, 203.457061215, 204.033984292, 204.616508563, 205.204743858, 205.798803264, 206.398803264, 207.00486387, 207.617108768, 208.235665469, 208.860665469, 209.492244416, 210.130542288]

        sr = 44100
        length = 215*sr
        impulseTrain = zeros(length)
        for beat in beats:
            impulseTrain[beat*sr]=1
        result = self.runInstance(impulseTrain)

        bpmList = [100, 140, 140, 100, 110, 110]
        expectedBpm = mean(bpmList)
        expectedBpmIntervals = [60./bpm for bpm in bpmList]
        expectedRubatoStart = [58.5, 179.31428571428467, 199.42617823213916]
        expectedRubatoStop = [61.157142857142958, 181.6714285714275, 201.75884724389928]

        #self.assertVectorWithinVector(result[0], bpmList, 1.)
        self.assertAlmostEqual(result[0], expectedBpm, 1.)

        # ticks
        self.assertVectorWithinVector(result[1], beats, 0.03)

        # bpm estimates
        self.assertVectorWithinVector(result[2], bpmList, 0.5)

        # bpm intervals
        self.assertVectorWithinVector(result[5], expectedBpmIntervals, 0.5)

        # rubato start/stop
        #self.assertAlmostEqualVector(result[3], expectedRubatoStart, .3)
        #self.assertAlmostEqualVector(result[4], expectedRubatoStop, .03)
        self.assertVectorWithinVector(result[3], expectedRubatoStart, 0.03)
        self.assertVectorWithinVector(result[4], expectedRubatoStop, 0.1)
    """


suite = allTests(TestRhythmExtractor)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
