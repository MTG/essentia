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
