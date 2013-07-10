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
import cmath # for asinh
import sys

class TestUtils(TestCase):
    def testIsSilent(self):
        self.assertEqual(True, isSilent([0]*100))

    def testInstantPower(self):
        sample = range(1,11)
        p = 0
        for s in sample:
            p += s**2
        p /= float( len(sample) )

        self.assertAlmostEqual(p, instantPower(sample))

    def testIsPowerOfTwo(self):
        self.assertTrue(isPowerTwo(0))
        top =131072
        k=1
        while k < top:
            self.assertTrue(isPowerTwo(k))
            k*=2
        while k < top:
            k=2*k+1
            self.assertTrue(not isPowerTwo(k))

    def testNextPowerOfTwo(self):
        self.assertEqual(nextPowerTwo(0), 0)
        self.assertEqual(nextPowerTwo(1), 1)
        top =131072
        k=2
        lastPowerTwo = 2
        while k < top:
            if not isPowerTwo(k): self.assertEqual(nextPowerTwo(k), 2*lastPowerTwo)
            else:
                self.assertEqual(nextPowerTwo(k), k)
                lastPowerTwo=k
            k +=1

    def testLinToDb(self):
        lin = 12.34
        expected_db = 10 * math.log10(lin)
        self.assertAlmostEqual(expected_db, lin2db(lin))

    def testDbToLin(self):
        db = -45.5
        expected_lin = 10**(db/10.)
        self.assertAlmostEqual(expected_lin, db2lin(db) , 5e-7)

    def testPowToDb(self):
        pow = 12.34
        expected_db = 10 * math.log10(pow)
        self.assertAlmostEqual(expected_db, pow2db(pow))

    def testDbToPow(self):
        db = -45.5
        expected_pow = 10**(db/10.)
        self.assertAlmostEqual(expected_pow, db2pow(db) , 5e-7)

    def testAmpToDb(self):
        amp = 12.34
        expected_db = 20 * math.log10(amp)
        self.assertAlmostEqual(expected_db, amp2db(amp))

    def testDbToAmp(self):
        db = -45.5
        expected_amp = 10**(0.5*db/10.)
        self.assertAlmostEqual(expected_amp, db2amp(db) , 5e-7)

    def testBarkToHz(self):
        bark = 5
        expected_hz = 1960 * (bark + 0.53) / (26.28 - bark)
        self.assertAlmostEqual(expected_hz, bark2hz(bark))

    def testHzToBark(self):
        hz = 440
        expected_bark = 26.81*hz / (1960 + hz) - 0.53
        self.assertAlmostEqual(expected_bark, hz2bark(hz))

    def testMelToHz(self):
        mel = 5
        expected_hz = 700.0*(math.exp(mel/1127.01048)-1.0)
        self.assertAlmostEqual(expected_hz, mel2hz(mel))

    def testHzToMel(self):
        hz = 440
        expected_mel = 1127.01048*math.log(hz/700.0+1.0)
        self.assertAlmostEqual(expected_mel, hz2mel(hz))


suite = allTests(TestUtils)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
