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

testdir = join(filedir(), 'windowing')

def hamming(size):
  window = []
  for i in range(size) :
    window.append(0.53836-0.46164*math.cos((2.0*math.pi*float(i))/(size-1.0)))
  return window

def hann(size):
  window = []
  for i in range(size) :
    window.append(0.5*(1-math.cos((2.0*math.pi*float(i))/(size-1.0))))
  return window

def triangular(size):
  window = []
  for i in range(size) :
    window.append(2.0/size*(size/2.0-abs(float(i-(size-1.0)/2.0))))
  return window

class TestWindowing(TestCase):

    def testSize(self):
        # This checks whether the output size of the windowed signal is as expected
        inputSize = 2047
        hintSize = 4095
        paddingSize = 16383 - hintSize

        input = [1] * inputSize
        output = Windowing(size=hintSize,  zeroPadding=paddingSize,  type='hann')(input)
        self.assertEqual(len(output), inputSize + paddingSize)

    def testZeropadding(self):
        # Checks whether the signal gets zero-padded correctly
        inputSize = 9
        halfInputSize = math.ceil(inputSize / 2.0)
        paddingSize = 9

        input = [1] * inputSize
        output = Windowing(size=inputSize,  zeroPadding=paddingSize,  type='square')(input)
        self.assertEqualVector([0] * paddingSize,  output[halfInputSize:halfInputSize + paddingSize])

    def testZero(self):
        # Checks whether the result of windowing a zero signal is zeros
        inputSize = 10
        paddingSize = 10

        input = [0] * inputSize
        output = Windowing(size=inputSize,  zeroPadding=paddingSize,  type='square')(input)
        self.assertEqualVector([0] * (inputSize + paddingSize),  output)

    def normalize(self, window = []):
        if window == None:
            return None
        sum_win = sum(window)
        return [2.0*i/sum_win for i in window]

    def testRegression(self):
        # Checks whether the windows are as expected
        inputSize = 1024
        input = [1] * inputSize

        # cannot use reference file as we use a different formula than others.
        # Essentia uses 0.53836 instead of 0.54 and 0.46164 instead of 0.46
        expected = self.normalize(hamming(inputSize))
        output = Windowing(size=inputSize,  zeroPadding=0,  type='hamming', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-6)

        expected = self.normalize(hann(inputSize))
        output = Windowing(size=inputSize,  zeroPadding=0,  type='hann', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-6)

        expected = self.normalize(triangular(inputSize))
        output = Windowing(size=inputSize,  zeroPadding=0,  type='triangular', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-6)

        expected = self.normalize(ones(inputSize))
        output = Windowing(size=inputSize,  zeroPadding=0,  type='square', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output)

        expected = self.normalize(readVector(join(testdir, str(inputSize),'blackmanharris62.txt')))
        output = Windowing(size=inputSize,  zeroPadding=0, type='blackmanharris62', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-5)

        expected = self.normalize(readVector(join(testdir, str(inputSize),'blackmanharris70.txt')))
        output = Windowing(size=inputSize,  zeroPadding=0, type='blackmanharris70', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-5)

        expected = self.normalize(readVector(join(testdir, str(inputSize),'blackmanharris74.txt')))
        output = Windowing(size=inputSize,  zeroPadding=0, type='blackmanharris74', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-5)

        expected = self.normalize(readVector(join(testdir, str(inputSize),'blackmanharris92.txt')))
        output = Windowing(size=inputSize,  zeroPadding=0, type='blackmanharris92', zeroPhase=False)(input)
        self.assertAlmostEqualVector(expected, output, 1e-3)

#        expected = readVector(join(testdir, 'hamming.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='hamming')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'hann.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='hann')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'triangular.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='triangular')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'square.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='square')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'blackmanharris62.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='blackmanharris62')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'blackmanharris70.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='blackmanharris70')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'blackmanharris74.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='blackmanharris74')(input)
#        self.assertAlmostEqualVector(expected, output)
#
#        expected = readVector(join(testdir, 'blackmanharris92.txt'))
#        output = Windowing(size=inputSize,  zeroPadding=0,  type='blackmanharris92')(input)
#        self.assertAlmostEqualVector(expected, output)

    def testEmpty(self):
        # Checks whether an empty input vector yields an exception
        self.assertComputeFails(Windowing(),  [])

    def testOne(self):
        # Checks for a single value
        self.assertComputeFails(Windowing(type='hann'),  [1])

    def testInvalidParam(self):
        self.assertConfigureFails(Windowing(), { 'type': 'unknown' })

suite = allTests(TestWindowing)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
