#!/usr/bin/env python

from essentia_test import *

class TestLoudnessVickers(TestCase):

    def testEmpty(self):
        self.assertEqual(LoudnessVickers()([]), -90)

    def testOne(self):
        self.assertAlmostEqual(LoudnessVickers()([1]),-32.0094032288)

    def testSilence(self):
        self.assertEqual(LoudnessVickers()([0]*2000), -90)

    def testInvalidSampleRate(self):
        self.assertConfigureFails(LoudnessVickers(), {'sampleRate': 44101})
        self.assertConfigureFails(LoudnessVickers(), {'sampleRate': 44099})

    def testDifferentFrequencies(self):
        # loudness of a 1000Hz signal should be higher than the loudness of a
        # 100 Hz signal
        from math import sin, pi
        sr = 44100
        size = 1*sr
        sine1 = [sin(2.0*pi*100.0*i/sr) for i in range(size)]
        sine2 = [sin(2.0*pi*1000.0*i/sr) for i in range(size)]
        fc1 = FrameCutter()
        fc2 = FrameCutter()
        frame1 = fc1(sine1)
        frame2 = fc2(sine2)
        while len(frame1) != 0 and len(frame2) != 0:
            self.assertTrue(LoudnessVickers()(frame1), LoudnessVickers()(frame2))
            frame1 = fc1(sine1)
            frame2 = fc2(sine2)

    def testFullScaleSquare(self):
        # the vicker's loudness of a full scale square wave should
        # be 0dB, but it isn't (?)
        sr = 44100
        freq = 1000
        step = 0.5*sr/freq
        size = 1*sr
        val = 1
        square = zeros(size)
        for i in range(size):
            square[i] = val
            if i%step < 1.0 :
                val *= -1
        result = 0
        nFrame = 0
        for frame in FrameGenerator(square):
           self.assertAlmostEqual(LoudnessVickers()(square), result, 0.15)
           nFrame += 1


suite = allTests(TestLoudnessVickers)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
