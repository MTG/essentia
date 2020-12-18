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
import math


class TestTempoTapMaxAgreement(TestCase):


    def testInvalidParam(self):
        self.assertConfigureFails(TempoTapMaxAgreement(), { 'minTempo': 181 })
        self.assertConfigureFails(TempoTapMaxAgreement(), { 'minTempo': 39 })
        self.assertConfigureFails(TempoTapMaxAgreement(), { 'maxTempo': 251 })
        self.assertConfigureFails(TempoTapMaxAgreement(), { 'maxTempo': 59 })
        self.assertConfigureFails(TempoTapMaxAgreement(), { 'minTempo': 160, 'maxTempo': 178 })


    def testRegression(self, tempotap = None):
        features = readMatrix(join(filedir(), 'tempotapmaxagreement', 'input.txt'))
        expected = readMatrix(join(filedir(), 'tempotapmaxagreement', 'output.txt'))

        numberFrames = 1024

        if not tempotapmaxagreement:
            tempotapmaxagreement = TempoTapMaxAgreement(tickCandidates = tickCandidates)

        tickEstimates = []
        confidenceEstimates = []

        for i in range(numberFrames*5):
            ticks, confidence = tempotapmaxagreement(features[i % len(features)])
            tickEstimates += list(ticks)
            confidenceEstimates += list(confidence)

        self.assertAlmostEqualVector(tickEstimates, expected[0], 1e-4)
        self.assertAlmostEqualVector(confidenceEstimates, expected[1], 1e-4)


    def testResetMethod(self):
        # NB: this test should actually fail when not calling reset, which it
        # doesn't at the moment...
        numberFrames = 1024

        tempotapmaxagreement = TempoTapMaxAgreement(tickCandidates = tickCandidates)
        
        self.testRegression()
        tempotapmaxagreement.reset()
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

        tempotapmaxagreement = TempoTapMaxAgreement(tickCandidates = tickCandidates)

        tickEstimates = []
        confidenceEstimates = []

        for i in range(inputLength):
            ticks, confidence = tempotapmaxagrement(features[i])

            tickEstimates += list(ticks)
            confidenceEstimates += list(confidence)

        # flush current buffer with zeros
        for i in range(inputLength % numberFrames, numberFrames):
            ticks, confidence = tempotapmaxagreement(zeros(featuresNumber))

            tickEstimates += list(ticks)
            confidenceEstimates += list(confidence)

        # TODO What is expected size
        # check that we have as many candidates as expected
        expectedSize = round(float(featuresNumber) * inputLength / numberFrames)
        self.assertEqual(len(tickEstimates), expectedSize)
        self.assertEqual(len(confidenceEstimates), expectedSize)
 
        # TODO : figure out what to do here
        # check that the first candidates fit with trial input
        #self.assertEqual(round(periodEstimates[0]), guessPeriod)
        #self.assertEqual(round(phaseEstimates[0]), guessPhase)

        # TODO in MaxAgreement
        # construct the beats train from the period and estimates candidates and
        # make sure we find a corresponding impulse in the input

    def testZero(self):
        tempotapmaxagreement = TempoTapMaxAgreement()

        tickEstimates = []
        confidenceEstimates = []

        for i in range(1300):
            ticks, confidence = tempotapmaxagreement(zeros(12))

            tickEstimates += list(ticks)
            confidenceEstimates += list(confidence)

        self.assert_(all(array(tickEstimates) == 0))
        # WTF?? that should probably raise an exception or sth...
        self.assert_(all(array(confidenceEstimates) == -1))



suite = allTests(TestTempoTapMaxAgreement)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
