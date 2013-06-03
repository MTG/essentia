#!/usr/bin/env python

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
