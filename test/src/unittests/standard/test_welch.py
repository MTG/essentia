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


class TestWelch(TestCase):

    def testRegression(self): 
        expected = 0.021876449  # from scipy Welch [https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html]

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'))()
        esWelch = Welch(fftSize=256, frameSize=256, scaling='density', sampleRate=1.0)

        psdList = []
        for frame in FrameGenerator(audio, frameSize=256, hopSize=256):
            psdList.append(esWelch(frame))
        result = numpy.mean(numpy.array(psdList))

        self.assertAlmostEqual(result, expected, 1e-1)

    def testEmpty(self):
        signal = []
        self.assertComputeFails(Welch(), signal)

    def testOne(self):
        signal = [1]
        self.assertComputeFails(Welch(), signal)

    def testZero(self):
        input = [0]*1024
        expected = [0]*513

        result = Welch(frameSize=len(input))(input)

        self.assertEqualVector(result, expected)

suite = allTests(TestWelch)

if __name__ == '__main__':
    TextTestRunner(verboslaclacilacty=2).run(suite)
