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


class TestGapsDetector(TestCase):

    def InitGapsDetector(self, **kwargs):
        return GapsDetector(**kwargs)

    def testZero(self):
        # inputting zeros should return an empty list.
        size = 1024
        self.assertEqualVector(
            self.InitGapsDetector(frameSize=size)(
                esarr(numpy.zeros(size)))[0], esarr([]))

    def testBadFrameSize(self):
        # if the frameSize is not properly set we throw an exception instead
        # of resizing as probably the hop size is mismatching too.
        sizes = [1024, 2047, 2049]  # default is 2048
        for size in sizes:
            self.assertComputeFails(
                GapsDetector(), esarr(numpy.zeros(size)))

    def testRegression(self, frameSize=2048, hopSize=1024):
        fs = 44100
        startsGroundTruth = [17.586]
        endsGroundTruth = [17.618]

        audio = MonoLoader(filename=join(testdata.audio_dir,
                           'recorded/mozart_c_major_30sec.wav'),
                           sampleRate=fs)()

        gapsDetector = GapsDetector(frameSize=frameSize, hopSize=hopSize)
        startsList = []
        endsList = []
        for frame in FrameGenerator(audio, frameSize=frameSize,
                                    hopSize=hopSize, startFromZero=True):
            starts, ends = gapsDetector(frame)

            if not len(starts) == 0:
                for start in starts:
                    startsList.append(start)
            if not len(ends) == 0:
                for end in ends:
                    endsList.append(end)
        self.assertAlmostEqualVector(startsList, startsGroundTruth, 1e-2)
        self.assertAlmostEqualVector(endsList, endsGroundTruth, 1e-2)

    def testNoOverlap(self):
        # the algorithm should also work without overlapping
        self.testRegression(frameSize=512, hopSize=512)

    def testInvalidParam(self):
        self.assertConfigureFails(GapsDetector(), {'sampleRate': -1.})
        self.assertConfigureFails(GapsDetector(), {'frameSize': 0})
        self.assertConfigureFails(GapsDetector(), {'hopSize': -1})
        self.assertConfigureFails(GapsDetector(), {'prepowerTime': -.5})
        self.assertConfigureFails(GapsDetector(), {'minimumTime': -.5})
        self.assertConfigureFails(GapsDetector(), {'maximumTime': -.5})
        self.assertConfigureFails(GapsDetector(), {'kernelSize': -1})
        self.assertConfigureFails(GapsDetector(), {'attackTime': -.5})
        self.assertConfigureFails(GapsDetector(), {'releaseTime': -.5})
        self.assertConfigureFails(GapsDetector(), {'frameSize': 64,
                                                   'hopSize': 32,
                                                   'kernelSize': 64})


suite = allTests(TestGapsDetector)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
