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
from numpy import sin, float32, pi, arange, mean, log2, floor, ceil

class TestVibrato(TestCase):

    def testEmpty(self):
        vibrato, extend = Vibrato()([])
        self.assertEqual(vibrato.size, 0)
        self.assertEqual(extend.size, 0)

    def testZero(self):
        vibrato, extend = Vibrato()([0])

        self.assertEqualVector(vibrato, [0.])
        self.assertEqualVector(extend, [0.])

    def testOnes(self):
        vibrato, extend = Vibrato()([1]*1024)

        self.assertEqualVector(vibrato, [0.]*1024)
        self.assertEqualVector(extend, [0.]*1024)


    def testNegativeInput(self):
        # Negative vlues should be set to 0
        vibrato, extend = Vibrato()([-1]*1024)

        self.assertEqualVector(vibrato, [0.]*1024)
        self.assertEqualVector(extend, [0.]*1024)    

    def testInvalidParam(self):
        self.assertConfigureFails(Vibrato(), { 'maxExtend':    -1 })
        self.assertConfigureFails(Vibrato(), { 'maxFrequency': -1 })
        self.assertConfigureFails(Vibrato(), { 'minExtend':    -1 })
        self.assertConfigureFails(Vibrato(), { 'minFrequency': -1 })
        self.assertConfigureFails(Vibrato(), { 'sampleRate':   -1 })

    def testSyntheticVibrato(self):
        fs = 100  #Hz
        f0 = 100  #Hz
        extend = 5  #Hz
        vibrato = 5 #Hz

        x = [f0] * 1024 + extend * sin(2 * pi * vibrato * arange(1024) / fs)

        vibratoEst, extendEst = Vibrato(sampleRate=fs)(x.astype(float32))

        self.assertAlmostEqual(vibrato,mean(vibratoEst[3:-3]), 1)

        extendCents = 1200*log2((f0+extend)/float(f0-extend))
        self.assertAlmostEqual(extendCents, mean(extendEst[3:-3]), 1e-1)

    def testFrequencyDetenctionThreshold(self):
        #  Vibrato Min and Max default values are 3 and 8Hz so
        #  this values shouldn't be detected.

        fs = 100  #Hz
        f0 = 100  #Hz
        extend = 5  #Hz
        vibrato = 3 #Hz
        x = [f0] * 1024 + extend * sin(2 * pi * vibrato * arange(1024) / fs)

        vibratoEst, extendEst = Vibrato(sampleRate=fs)(x.astype(float32))
        self.assertEqual(0, vibratoEst.all())

        vibrato = 9 #Hz
        x = [f0] * 1024 + extend * sin(2 * pi * vibrato * arange(1024) / fs)
        vibratoEst, extendEst = Vibrato(sampleRate=fs)(x.astype(float32))
        self.assertEqual(0, vibratoEst.all())


    def testExtendDetenctionThreshold(self):
        #  Extend Min and Max default values are 50 and 250 cents so
        #  this values  shouldn't be detected.
        fs = 100  #Hz
        f0 = 100  #Hz
        extend = ceil(f0 * (2**(250/1200.) -1) / (2**(250/1200.) + 1))
        vibrato = 5 #Hz
        x = [f0] * 1024 + extend * sin(2 * pi * vibrato * arange(1024) / fs)
        vibratoEst, extendEst = Vibrato(sampleRate=fs)(x.astype(float32))
        self.assertEqual(0, vibratoEst.all())

        extend = floor(f0 * (2**(50/1200.) -1) / (2**(50/1200.) + 1))
        x = [f0] * 1024 + extend * sin(2 * pi * vibrato * arange(1024) / fs)
        vibratoEst, extendEst = Vibrato(sampleRate=fs)(x.astype(float32))
        self.assertEqual(0, vibratoEst.all())


suite = allTests(TestVibrato)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)