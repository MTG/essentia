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
import essentia
from essentia import Pool
import essentia.standard as std
from essentia.streaming import *

from math import ceil, fabs
from numpy import argmax


class TestBpmHistogram(TestCase):

    def computeNoveltyCurve(self, filename, frameSize=1024, hopSize=512, windowType='hann',
                            weightCurveType='inverse_quadratic', sampleRate=44100.0,
                            startTime=0, endTime=2000):

        loader = EasyLoader(filename=filename, startTime=startTime,
                            endTime=endTime, sampleRate=sampleRate,
                            downmix='left')
        fc     = FrameCutter(frameSize=frameSize, hopSize=hopSize,
                             silentFrames="keep",
                             startFromZero=False, lastFrameToEndOfFile=True)
        window = Windowing(type=windowType, zeroPhase=True,
                           zeroPadding=1024-frameSize)
        freqBands = FrequencyBands(sampleRate=sampleRate) # using barkbands by default
        pool = Pool()
        spec = Spectrum()
        loader.audio >> fc.signal
        fc.frame >> window.frame >> spec.frame
        spec.spectrum >> freqBands.spectrum
        freqBands.bands >> (pool, 'frequency_bands')
        essentia.run(loader)


        noveltyCurve = std.NoveltyCurve(frameRate=sampleRate/float(hopSize),
                                        weightCurveType=weightCurveType)(pool['frequency_bands'])

        return noveltyCurve

    def computeBpmHistogram(self, noveltyCurve, frameSize=4, overlap=2,
                            frameRate=44100./128., window='hann',
                            zeroPadding = 0,
                            constantTempo=False):

        pool=Pool()
        bpmHist = BpmHistogram(frameRate=frameRate,
                               frameSize=frameSize,
                               overlap=overlap,
                               zeroPadding=zeroPadding,
                               constantTempo=constantTempo,
                               windowType='hann')

        gen    = VectorInput(noveltyCurve)
        gen.data >> bpmHist.novelty
        bpmHist.bpm >> (pool, 'bpm')
        bpmHist.bpmCandidates >> (pool, 'bpmCandidates')
        bpmHist.bpmMagnitudes >> (pool, 'bpmMagnitudes')
        bpmHist.frameBpms >> None #(pool, 'frameBpms')
        bpmHist.tempogram >> (pool, 'tempogram')
        bpmHist.ticks >> (pool, 'ticks')
        bpmHist.ticksMagnitude >> (pool, 'ticksMagnitude')
        bpmHist.sinusoid >> (pool, 'sinusoid')
        essentia.run(gen)

        return pool

    def testSyntheticNoveltyCurve(self):
        # in this test we assume the noveltyCurve is a perfect sine. The algorithm
        # should reproduce a sinusoid with peaks at the same positions as those
        # from the fake noveltycurve. Ploting may help to understand...
        frameSize = 4 # 4 seconds
        overlap = 2
        frameRate = 44100/128.
        f = 10 # Hz
        f = f*float(frameRate)/nextPowerTwo(int(ceil(frameSize*frameRate)))
        expectedBpm = f*60. # 101Bpm

        length = 3000
        noveltyCurve = [numpy.sin(2.0*numpy.pi*f*x/frameRate + 0.135*numpy.pi) for x in range(length)]
        for idx,x in enumerate(noveltyCurve):
            if x<0: noveltyCurve[idx] = 0
        pool = self.computeBpmHistogram(noveltyCurve, frameSize, overlap, frameRate)

        #plot(noveltyCurve)
        #plot(pool['sinusoid'],'r')
        #show()

        noveltyPeaks = std.PeakDetection(interpolate=False)(noveltyCurve)[0]
        sinusoidPeaks = std.PeakDetection(interpolate=False)(pool['sinusoid'])[0]

        # depending on the framesize, hopsize, etc. the sinusoid is usually
        # larger than the novelty curve, so we need to trim the sinusoid's
        # peaks
        sinusoidPeaks = std.PeakDetection()(pool['sinusoid'])[0][:len(noveltyPeaks)]
        for p1, p2 in zip(noveltyPeaks, sinusoidPeaks):
            self.assertAlmostEqual(fabs(p1-p2), 0, 5e-2)
        self.assertAlmostEqual(pool['bpm'], expectedBpm, 1e-3)


    def testZero(self):
        # zeros should return no onsets (empty array)
        pool = self.computeBpmHistogram(zeros(10*44100))
        self.assertEqual(pool['bpm'], 0)
        self.assertEqualVector(pool['sinusoid'], [])
        self.assertEqualVector(pool['ticks'], [])


    def testConstantInput(self):
        # constant input reports bpm 0
        pool = self.computeBpmHistogram(ones(10*44100))
        self.assertEqual(int(pool['bpm']), 0)
        self.assertEqual(sum(pool['bpmCandidates']), 0)
        self.assertEqual(sum(pool['bpmMagnitudes']), 0)



    def testRegressionVariableTempoImpulse(self):
        # file with a variable click track:
        # 0-20s: 120bpm
        # 20-40s: 240bpm
        # 20-60s: 180bpm
        filename = join(testdata.audio_dir, 'generated', 'synthesised', 'beat_120_240_180.wav')
        pool = self.computeBpmHistogram(self.computeNoveltyCurve(filename),
                                        zeroPadding = 1,
                                        frameRate=44100./512.)
        # as 120 and 240 are harmonics, the algorithm will just throw the
        # lowest, i.e. 120 and also 180 as 120 are not pure harmonics, although
        # they are both harmonics of 60...
        expectedBpm=240
        self.assertAlmostEqual(int(pool['bpm']), expectedBpm, 1)

        # test that all candidates are multiples of 60:
        for bpm in pool['bpmCandidates']:
            k = bpm/60.
            self.assertTrue(fabs(k-int(k)) < 0.1)

        # check that the tempogram reflects the variation of the tempo:
        expectedMaxTempos = [120,120,120,120,120,120,120,
                             240,240,240,240,240,240,240,
                             180,180,180,180,180,180,180]
        maxTempos = []
        for tempogram in pool['tempogram'][0]:
            maxTempos.append(argmax(tempogram))
        self.assertAlmostEqualVector(maxTempos, expectedMaxTempos, 1);


    def testRegressionReal(self):
        filename =filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        pool = self.computeBpmHistogram(self.computeNoveltyCurve(filename),
                                        frameSize=4, overlap=2,
                                        frameRate=44100./512.)
        expectedBpm = 190 # could also be 95-96
        self.assertEqual(int(pool['bpm']), expectedBpm)
        # the ticks that are being tested with were not manually annotated.
        # It's just the output of the algorithm written to a file
        # Therefore this test is just for checking that everything stays as originally:
        # the algo outputs around 104 ticks. We should get exactly
        # expectedBpm/2, but this can only happen when the algo is run
        # recursively, as is done in src/examples/streaming_beattrack.cpp
        referenceTicks = readVector(join(filedir(), 'bpmhistogram/britney.beat'))
        ticks = list(pool['ticks'])
        while ticks[-1] > 30: ticks.pop() # we get ticks beyond the audio file
        self.assertEqual(len(ticks), expectedBpm/2)
        self.assertAlmostEqualVector(ticks, referenceTicks, 1e-2)

    def testEmpty(self):
        # nothing should be computed and the resulting pool be empty
        pool = self.computeBpmHistogram([])
        self.assertEqualVector(pool.descriptorNames(), [])


suite = allTests(TestBpmHistogram)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
