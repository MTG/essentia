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
import math


class TestTempoTap(TestCase):


    def testInvalidParam(self):
        self.assertConfigureFails(TempoTap(), { 'minTempo': 181 })
        self.assertConfigureFails(TempoTap(), { 'minTempo': 39 })
        self.assertConfigureFails(TempoTap(), { 'maxTempo': 251 })
        self.assertConfigureFails(TempoTap(), { 'maxTempo': 59 })
        self.assertConfigureFails(TempoTap(), { 'minTempo': 160, 'maxTempo': 178 })


    def testRegression(self, tempotap = None):
        features = readMatrix(join(filedir(), 'tempotap', 'input.txt'))
        expected = readMatrix(join(filedir(), 'tempotap', 'output.txt'))

        numberFrames = 1024

        if not tempotap:
            tempotap = TempoTap(numberFrames = numberFrames,
                                frameHop = numberFrames)

        periodEstimates = []
        phaseEstimates = []

        for i in range(numberFrames*5):
            periods, phases = tempotap(features[i % len(features)])
            periodEstimates += list(periods)
            phaseEstimates += list(phases)

        self.assertAlmostEqualVector(periodEstimates, expected[0], 1e-4)
        self.assertAlmostEqualVector(phaseEstimates, expected[1], 1e-4)


    def testResetMethod(self):
        # NB: this test should actually fail when not calling reset, which it
        # doesn't at the moment...
        numberFrames = 1024
        tempotap = TempoTap(numberFrames = numberFrames,
                            frameHop = numberFrames)

        self.testRegression()
        tempotap.reset()
        self.testRegression()


    def testImpulseTrain(self):
        inputLength = 14200
        numberFrames = 1024
        featuresNumber = 1
        guessPeriod = 70
        guessPhase = 17

        # create an impulse train of phase 17 and period 70
        features = zeros((inputLength, featuresNumber))
        i = int(math.floor(guessPhase))
        while i < len(features):
            features[i][0] = 1
            i += int(math.floor(guessPeriod))


        tempotap = TempoTap(numberFrames = numberFrames,
                            frameHop = numberFrames)

        periodEstimates = []
        phaseEstimates = []

        for i in range(inputLength):
            periods, phases = tempotap(features[i])

            periodEstimates += list(periods)
            phaseEstimates += list(phases)

        # flush current buffer with zeros
        for i in range(inputLength % numberFrames, numberFrames):
            periods, phases = tempotap(zeros(featuresNumber))

            periodEstimates += list(periods)
            phaseEstimates += list(phases)


        # check that we have as many candidates as expected
        expectedSize = round(float(featuresNumber) * inputLength / numberFrames)
        self.assertEqual(len(periodEstimates), expectedSize)
        self.assertEqual(len(phaseEstimates), expectedSize)

        # check that the first candidates fit with trial input
        self.assertEqual(round(periodEstimates[0]), guessPeriod)
        self.assertEqual(round(phaseEstimates[0]), guessPhase)


        # construct the beats train from the period and estimates candidates and
        # make sure we find a corresponding impulse in the input
        i = 0
        while i < len(phaseEstimates):
            while phaseEstimates[i] < numberFrames:
                currentBeat = round(phaseEstimates[i] + i*numberFrames)
                if (currentBeat > inputLength):
                    break
                self.assertEqual(features[int(currentBeat)][0], 1)
                phaseEstimates[i] += round(periodEstimates[i])
            i += 1



    def testZero(self):
        tempotap = TempoTap()

        periodEstimates = []
        phaseEstimates = []

        for i in range(1300):
            periods, phases = tempotap(zeros(12))

            periodEstimates += list(periods)
            phaseEstimates += list(phases)

        self.assert_(all(array(periodEstimates) == 0))
        # WTF?? that should probably raise an exception or sth...
        self.assert_(all(array(phaseEstimates) == -1))



suite = allTests(TestTempoTap)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
