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
import sys
from numpy import pi, sin
import os

class TestMonoWriter(TestCase):

    def testRegression(self):
        sr = 44100
        inputSize = sr # 1 second of audio
        input = [0.5*sin(2.0*pi*440.0*i/inputSize) for i in range(inputSize)]
        MonoWriter(filename = "foo.wav", sampleRate = sr)(input)
        left = MonoLoader(filename = 'foo.wav', downmix='left', sampleRate=sr)()
        os.remove('foo.wav')
        self.assertAlmostEqualVector(left, input, 5e-2)

        sr = 48000
        inputSize = sr # 1 second of audio
        input = [0.5*sin(2.0*pi*440.0*i/inputSize) for i in range(inputSize)]
        MonoWriter(filename = "foo.wav", sampleRate = sr)(input)
        left = MonoLoader(filename = 'foo.wav', downmix='left', sampleRate=sr)()
        os.remove('foo.wav')
        self.assertAlmostEqualVector(left, input, 5e-2)

        sr =22050
        inputSize = sr # 1 second of audio
        input = [0.5*sin(2.0*pi*440.0*i/inputSize) for i in range(inputSize)]
        MonoWriter(filename = "foo.wav", sampleRate = sr)(input)
        left = MonoLoader(filename = 'foo.wav', downmix='left', sampleRate=sr)()
        os.remove('foo.wav')
        self.assertAlmostEqualVector(left, input, 5e-2)


    def testEmpty(self):
        MonoWriter(filename = 'foo.wav')([])
        self.assertTrue( not os.path.exists('foo.wav') )

    def testInvalidParam(self):
        self.assertConfigureFails(MonoWriter(), { 'filename' : 'foo.wav', 'sampleRate' : 0 })

    def testInvalidFilename(self):
        self.assertComputeFails(MonoWriter(filename=''),([0.1, 0.2, 0.3]))

suite = allTests(TestMonoWriter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
