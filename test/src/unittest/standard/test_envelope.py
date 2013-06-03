#!/usr/bin/env python

from essentia_test import *
import numpy

class TestEnvelope(TestCase):

    def testFile(self):
        filename=join(testdata.audio_dir, 'generated', 'synthesised', 'sin_pattern_decreasing.wav')
        audioLeft = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        envelope = Envelope(sampleRate=44100, attackTime=5, releaseTime=100)(audioLeft)
        for x in envelope:
            self.assertValidNumber(x)

    def testEmpty(self):
        self.assertEqualVector(Envelope()([]), [])

    def testZero(self):
        input = [0]*100000
        envelope = Envelope(sampleRate=44100, attackTime=5, releaseTime=100)(input)
        self.assertEqualVector(envelope, input)

    def testOne(self):
        input = [-0.5]
        envelope = Envelope(sampleRate=44100, attackTime=0, releaseTime=100, applyRectification=True)(input)
        self.assertEqual(envelope[0], -input[0])

    def testInvalidParam(self):
        self.assertConfigureFails(Envelope(), { 'sampleRate': 0 })
        self.assertConfigureFails(Envelope(), { 'attackTime': -10 })
        self.assertConfigureFails(Envelope(), { 'releaseTime': -10 })

suite = allTests(TestEnvelope)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
