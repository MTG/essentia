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



# NOTE: frequencyBands = [20,150] is used throughout the tests in order to
# obtain the same results as before adding frequencyBands parameter to the
# beatsloudness algorithm

from essentia_test import *
from essentia.streaming import BeatsLoudness, MonoLoader as sMonoLoader, \
                               RhythmExtractor

class TestBeatsLoudness(TestCase):

    def computeSingleBeatLoudness(self, beat, audio, sr):
        beatWindowDuration = 0.1
        beatDuration = 0.05
        start = (beat - beatWindowDuration/2)*sr
        end = (beat + beatWindowDuration/2 + beatDuration + 0.0001)*sr
        # SingleBeatLoudness will throw exception if the audio fragment is too short,
        # this will happen when the beat is too close to the beginning of the signal so that 
        # the beat window will start actually before it
        if start < 0:
            # reposition the window
            end = start - end
            start = 0
        return SingleBeatLoudness(frequencyBands = [20,150])(audio[start:end])

    def testEmpty(self):
        gen = VectorInput([])
        beatsLoudness = BeatsLoudness()
        p = Pool()

        gen.data >> beatsLoudness.signal
        beatsLoudness.loudness >> (p, 'beats.loudness')
        beatsLoudness.loudnessBandRatio >> (p, 'beats.loudnessBandRatio')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testRegression(self):
        loader = sMonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'),
                             sampleRate=44100)
        rhythm = RhythmExtractor()
        p = Pool()

        loader.audio >> rhythm.signal
        loader.audio >> (p, 'audio.raw')
        rhythm.bpm >> None
        rhythm.bpmIntervals >> None
        rhythm.estimates >> None
        #rhythm.rubatoStart >> None
        #rhythm.rubatoStop >> None
        rhythm.ticks >> (p, 'beats.locationEstimates')

        run(loader)

        gen = VectorInput(p['audio.raw'])

        beatsLoudness = BeatsLoudness(beats=p['beats.locationEstimates'],
                                      frequencyBands = [20,150])

        gen.data >> beatsLoudness.signal
        beatsLoudness.loudness >> (p, 'beats.loudness')
        beatsLoudness.loudnessBandRatio >> (p, 'beats.loudnessBandRatio')
        run(gen);

        expectedLoudness = []
        expectedLoudnessBandRatio = []
        for beat in p['beats.locationEstimates']:
            loudness, loudnessBandRatio = self.computeSingleBeatLoudness(beat,p['audio.raw'], 44100)
            expectedLoudness.append(loudness)
            expectedLoudnessBandRatio.append(loudnessBandRatio)

       # The values below where extracted from running essentia-1.0  cpp tests
       # on some platform. This results cause the test to fail, and there is no
       # way to be sure they are correct. Therefore a new test has been done
       # where we compare the results of the algorithm with a manually passing
       # the beats to singlebeatloudness std:

       # expectedLoudness = [0.428758, 0.291341, 0.633762, 0.26555, 0.425245, 0.277024, 0.495149, 0.242385, 0.357601, 0.334, 0.323821, 0.232946, 0.528381, 0.200571, 0.437708, 0.167769, 0.584228, 0.392591, 0.530719, 0.296724, 0.550218, 0.332743, 0.501887, 0.310001, 0.403775, 0.29342, 0.578137, 0.306543, 0.470718, 0.690108, 0.0089495, 0.372516, 0.180331, 0.253785, 0.298147, 0.290077, 0.447453, 0.536407, 0.257739, 0.587473, 0.526467, 0.415834, 0.259945, 0.48784, 0.440733, 0.462674, 0.279204]
       # expectedLoudnessBass = [0.928696, 0.127746, 0.681139, 0.0506813, 0.947531, 0.0654974, 0.822909, 0.0516866, 0.781132, 0.134502, 0.74214, 0.0559918, 0.870337, 0.0795841, 0.825638, 0.0935618, 0.875636, 0.11054, 0.515007, 0.0459782, 0.681463, 0.0269587, 0.755229, 0.0620431, 0.711997, 0.127048, 0.713851, 0.0255558, 0.700511, 0.754544, 0.452143, 0.745394, 0.0926197, 0.113369, 0.0516325, 0.0871752, 0.00407939, 0.779901, 0.0498086, 0.677019, 0.0714908, 0.368265, 0.0453059, 0.51892, 0.0210914, 0.63086, 0.069424]


        self.assertAlmostEqualVector(p['beats.loudness'], expectedLoudness)
        self.assertAlmostEqualVector(p['beats.loudnessBandRatio'], expectedLoudnessBandRatio)


    def testClickTrack(self):
        sr = 44100
        nClicks = 5

        # create audio signal that represents a click track nClicks seconds
        # long, with .25s clicks starting at every second
        clickTrack = [1.0]*(sr/2) + [0.0]*(sr/2)
        clickTrack *= nClicks
        clickLocations = [i + 1./4. for i in range(nClicks)]

        gen = VectorInput(clickTrack)
        beatsLoudness = BeatsLoudness(beats=clickLocations,
                                      frequencyBands = [20,150],
                                      beatWindowDuration=0.5,
                                      beatDuration=0.5)

        p = Pool()

        gen.data >> beatsLoudness.signal
        beatsLoudness.loudness >> (p, 'beats.loudness')
        beatsLoudness.loudnessBandRatio >> (p, 'beats.loudnessBandRatio')

        run(gen)

        # last beat gets discarded as it cannot be completely acquired, thus
        # (nclicks-1)
        expectedLoudness = [5.22167]*(nClicks-1)
        expectedLoudnessBandRatio = [2.07204e-13]*(nClicks-1)

        self.assertAlmostEqualVector(p['beats.loudness'], expectedLoudness, 1e-5)
        self.assertAlmostEqualVector(p['beats.loudnessBandRatio'], expectedLoudnessBandRatio, 2e-2)


    def testLastBeatTooShort(self):
        beatDuration = 0.5 # in seconds

        # 1 second silence, beat for beatDuration (s), 1 second silence, then last beat lasts half beatDuration
        signal = [0]*44100 + [1]*int(44100*beatDuration) + [0]*44100 + [1]*int(44100*beatDuration/2)

        beatPositions = [1+beatDuration/2.0, 2.05+beatDuration/4.0] # each in seconds

        gen = VectorInput(signal)
        beatsLoudness = BeatsLoudness(beats=beatPositions,
                                      frequencyBands = [20,150],
                                      beatDuration=beatDuration,
                                      beatWindowDuration=0.5)
        p = Pool()

        gen.data >> beatsLoudness.signal
        beatsLoudness.loudness >> (p, 'beats.loudness')
        beatsLoudness.loudnessBandRatio >> (p, 'beats.loudnessBandRatio')

        run(gen)

        # the last beat should have been thrown away since it didn't last for a
        # whole beat duration
        self.assertAlmostEqualVector(p['beats.loudness'], [5.22167], 1e-5)
        self.assertAlmostEqualVector(p['beats.loudnessBandRatio'], [2.07204e-13], 2e-2)



suite = allTests(TestBeatsLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
