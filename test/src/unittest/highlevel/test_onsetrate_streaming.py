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
from essentia.streaming import OnsetRate

class TestStreamingOnsetRate(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        sor = OnsetRate()
        p = Pool()

        gen.data >> sor.signal
        sor.onsetTimes >> (p, 'data')
        sor.onsetRate >> (p, 'onset.rate')

        run(gen)

        self.assertEqual(len(p.descriptorNames()), 0)


    def testZero(self):
        gen = VectorInput( [0]*10*1024 )
        sor = OnsetRate()
        p = Pool()

        gen.data >> sor.signal
        sor.onsetTimes >> (p, 'onset.times')
        sor.onsetRate >> (p, 'onset.rate')

        run(gen)

        self.assertEqual(p['onset.rate'], 0)


    def ImpulseTrain(self, frameSize, factor, precision):
        nFrames = 16
        inputData = [0]*nFrames*frameSize
        pos = factor*frameSize

        # after the first frame,every factor frames there will be an impulse of
        # type 1, 0.8, 0.6, 0.4, 0.2
        for i in xrange(len(inputData)):
            mod = i%pos
            if i > frameSize and mod < 5:
                inputData[i] = 0.5*(5-mod)/2.5 # impulse: 1, 0.8, 0.6, 0.4, 0.2

        size = int(nFrames/factor);
        expected = [0]*size
        for i in xrange(size):
          expected[i] = factor*(i+1)*frameSize/44100

        gen = VectorInput(inputData)
        onsetRate = OnsetRate()
        p = Pool()

        gen.data >> onsetRate.signal
        onsetRate.onsetTimes >> (p, 'onset.times')
        onsetRate.onsetRate >> (p, 'onset.rate')

        run(gen)

        self.assertAlmostEqual(p['onset.rate'], len(expected)/float(len(inputData))/44100.0, 0.05)

        self.assertAlmostEqualVector(p['onset.times'], expected, precision)


    def testImpulseTrain(self):
        for i in range(4):
            frameSize = 2**i * 1024

            for j in range(20):
                factor = .05*j + 5

                self.ImpulseTrain(frameSize, factor, 0.1)


suite = allTests(TestStreamingOnsetRate)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
