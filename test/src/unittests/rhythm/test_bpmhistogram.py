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
import essentia
from essentia import Pool
from essentia.standard import *
import essentia.streaming as ess

from math import ceil, fabs
from numpy import argmax


class TestBpmHistogram(TestCase):

    def computeNoveltyCurve(self, filename, frameSize=1024, hopSize=512, windowType='hann',
                            weightCurveType='hybrid', sampleRate=44100.0,
                            startTime=0, endTime=2000):

        loader = ess.EasyLoader(filename=filename, startTime=startTime,
                                endTime=endTime, sampleRate=sampleRate,
                                downmix='left')
        fc     = ess.FrameCutter(frameSize=frameSize, hopSize=hopSize,
                                 silentFrames="keep",
                                 startFromZero=False, lastFrameToEndOfFile=True)
        window = ess.Windowing(type=windowType, zeroPhase=True,
                               zeroPadding=1024-frameSize)
        freqBands = ess.FrequencyBands(sampleRate=sampleRate) # using barkbands by default
        spec = ess.Spectrum()

        pool = Pool()
        loader.audio >> fc.signal
        fc.frame >> window.frame >> spec.frame
        spec.spectrum >> freqBands.spectrum
        freqBands.bands >> (pool, 'frequency_bands')
        essentia.run(loader)


        noveltyCurve = NoveltyCurve(frameRate=sampleRate/float(hopSize),
                                    weightCurveType=weightCurveType)(pool['frequency_bands'])

        return noveltyCurve

    def computeBpmHistogram(self, noveltyCurve, frameSize=4, overlap=2,
                            frameRate=44100./128., window='hann',
                            zeroPadding=0,
                            constantTempo=False,
                            minBpm=30):

        pool=Pool()
        bpmHist = ess.BpmHistogram(frameRate=frameRate,
                                   frameSize=frameSize,
                                   overlap=overlap,
                                   zeroPadding=zeroPadding,
                                   constantTempo=constantTempo,
                                   windowType='hann',
                                   minBpm=minBpm)

        gen    = ess.VectorInput(noveltyCurve)
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

        noveltyPeaks = PeakDetection(interpolate=False)(noveltyCurve)[0]
        sinusoidPeaks = PeakDetection(interpolate=False)(pool['sinusoid'])[0]

        # depending on the framesize, hopsize, etc. the sinusoid is usually
        # larger than the novelty curve, so we need to trim the sinusoid's
        # peaks
        sinusoidPeaks = PeakDetection()(pool['sinusoid'])[0][:len(noveltyPeaks)]
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
        pool = self.computeBpmHistogram(ones(10*44100), minBpm=0)
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
        # TODO: this test on a real example is not ready yet.
        # We need to add a song from Public Domain with beat annotation done by
        # expert and compare output against it"

        filename =filename = join(testdata.audio_dir, 'recorded', 'techno_loop.wav')
        pool = self.computeBpmHistogram(self.computeNoveltyCurve(filename),
                                        frameSize=4, overlap=2,
                                        frameRate=44100./512.)

        # Octave error, should actually be a double of it.
        # Also, the exact value should be 61.9
        expectedBpm = 61

        self.assertEqual(int(pool['bpm']), expectedBpm)
        # the ticks that are being tested with were not manually annotated.
        # It's just the output of the algorithm written to a file
        # Therefore this test is just for checking that everything stays as originally:
        # the algo outputs more ticks than exactly expectedBpm/2. The exact number can
        # only happen when the algo is run recursively, as is done in src/examples/streaming_beattrack.cpp
        referenceTicks = [0.49410257, 1.45644403, 2.42204642, 3.38822556, 4.35157299,
                          5.31459808, 6.28024578, 7.24001789,  8.20012093, 9.16794968,
                          10.12917042, 11.08399296, 12.05328083, 13.01444721, 13.96003532,
                          14.92348099, 15.88770294, 16.84051323, 17.80594826, 18.77087402,
                          19.7209034, 20.68716812, 21.65229988, 22.60784912, 23.57401085,
                          24.5430603, 25.48790741, 26.44778061, 27.41486359, 28.36922264,
                          29.33124924]  #, 30.30256462, 31.2541275, 32.20780945]

        ticks = list(pool['ticks'])
        while ticks[-1] > 30: ticks.pop()  # audio duration is 30 seconds, we get ticks beyond that

        self.assertEqual(len(ticks), 31)      
        # round(expectedBpm/2.) = 30 instead of 31 in python3
        #self.assertEqual(len(ticks), round(expectedBpm/2.))

        self.assertAlmostEqualVector(ticks, referenceTicks, 1e-2)

    def testEmpty(self):
        # nothing should be computed and the resulting pool be empty
        pool = self.computeBpmHistogram([])
        self.assertEqualVector(pool.descriptorNames(), [])


suite = allTests(TestBpmHistogram)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
