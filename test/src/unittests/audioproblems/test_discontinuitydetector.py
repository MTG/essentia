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
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/


from essentia_test import *
from essentia import array as esarr


class TestDiscontinuityDetector(TestCase):

    def InitDiscontinuityDetector(self, **kwargs):
        return DiscontinuityDetector(**kwargs)

    def testZero(self):
        # Inputting zeros should return zero.
        size = 1024
        self.assertEqualVector(
            self.InitDiscontinuityDetector(frameSize=size)(
                esarr(numpy.zeros(size)))[0], esarr([]))

    # The algorithm should be robust to squarewaves if there are at least a few
    # periods on the frame. ( ~200Hz for 512 samples @ 44.1kHz) Try different
    # frequencies.
    def testSquareWave(self):
        fs = 44100
        minFreq = 200  # Hz
        maxFreq = 20000  # Hz
        time = 10  # s

        for f in numpy.linspace(minFreq, maxFreq, 5):
            samplenum = int(fs / f)
            samplenum -= samplenum % 2

            waveTable = [0] * samplenum
            waveTable[:samplenum / 2] = [1] * (samplenum / 2)

            waveDur = len(waveTable) / 44100.
            repetitions = int(time / waveDur)
            input = waveTable * repetitions
            self.assertEqualVector(
                self.InitDiscontinuityDetector()(esarr(input))[0], esarr([]))
                

    def testInvalidParam(self):
        self.assertConfigureFails(DiscontinuityDetector(), {'order': 0})
        self.assertConfigureFails(DiscontinuityDetector(), {'frameSize': 0})
        self.assertConfigureFails(DiscontinuityDetector(), {'hopSize': 1024})
        self.assertConfigureFails(DiscontinuityDetector(), {'kernelSize': -1})
        self.assertConfigureFails(DiscontinuityDetector(), {'hopSize': 1024})
        self.assertConfigureFails(DiscontinuityDetector(), {'detectionThreshold': -12})
        self.assertConfigureFails(DiscontinuityDetector(), {'subFrameSize': 1024})
        self.assertConfigureFails(DiscontinuityDetector(), {'frameSize': 64,
                                                            'hopSize': 32,
                                                            'order': 64})


suite = allTests(TestDiscontinuityDetector)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
