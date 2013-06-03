#!/usr/bin/env python

from essentia_test import *


class TestAttackTime(TestCase):

    def setUp(self):
        self.envelope = Envelope(sampleRate = 44100,
                                 attackTime = 10.0,
                                 releaseTime = 10.0)

        self.attackTime = LogAttackTime(sampleRate = 44100,
                                        startAttackThreshold = 0.2,
                                        stopAttackThreshold = 0.9)


    def testFile(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/roxette.wav'),
                           sampleRate = 44100)()

        fc = FrameCutter(frameSize = 1024, hopSize = 512)

        while True:
            frame = fc(audio)

            if len(frame) == 0:
                break

            atime = self.attackTime(self.envelope(frame))

            self.assert_(not numpy.isinf(atime))
            self.assert_(not numpy.isnan(atime))



    def testZero(self):
        smoothed = self.envelope(zeros(1024))
        self.assert_((smoothed == zeros(1024)).all())

        atime = self.attackTime(smoothed)
        self.assertEqual(atime, -5.0)


suite = allTests(TestAttackTime)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
