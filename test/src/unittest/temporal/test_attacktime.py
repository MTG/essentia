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


class TestAttackTime(TestCase):

    def setUp(self):
        self.envelope = Envelope(sampleRate = 44100,
                                 attackTime = 10.0,
                                 releaseTime = 10.0)

        self.attackTime = LogAttackTime(sampleRate = 44100,
                                        startAttackThreshold = 0.2,
                                        stopAttackThreshold = 0.9)


    def testFile(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/musicbox.wav'),
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
