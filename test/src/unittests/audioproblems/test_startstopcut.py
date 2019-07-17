#!/usr/bin/env python

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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


class TestStartStopCut(TestCase):
    def InitStartStopCut(self, **kwargs):
        return StartStopCut(**kwargs)

    def testZero(self):
        # Test different input sizes.
        size = 200000  # apx. 4.5s @ 44.1kHz
        while size > 1000:
            self.assertEqualVector(
                self.InitStartStopCut()(
                    esarr(numpy.zeros(size))), (0, 0))
            size //= 2

    def testOnes(self):
        size = 200000  # apx. 4.5s @ 44.1kHz
        while size > 1000:
            self.assertEqualVector(
                self.InitStartStopCut()(
                    esarr(numpy.ones(size))), (1, 1))
            size //= 2

    def testInputTooShort(self):
        # If the input size is smaller that the detection thresholds plus
        # the size of a frame it should throw an Exception.
        size = 1024
        self.assertComputeFails(
            StartStopCut(frameSize=size), esarr(numpy.ones(size)))

    def testRegression(self, frameSize=512, hopSize=256):
        fs = 44100.
        audio = MonoLoader(filename=join(testdata.audio_dir,
                           'recorded/mozart_c_major_30sec.wav'),
                           sampleRate=fs)()

        startStopCut = StartStopCut(frameSize=frameSize, hopSize=hopSize)

        start, end = startStopCut(audio)

        self.assertEqual(start, 0)
        self.assertEqual(end, 1)

    def testRegressionNoOverlap(self):
        self.testRegression(frameSize=256, hopSize=256)

    def testInvalidParam(self):
        self.assertConfigureFails(StartStopCut(), {'sampleRate': -1.})
        self.assertConfigureFails(StartStopCut(), {'frameSize': 0})
        self.assertConfigureFails(StartStopCut(), {'hopSize': -1})
        self.assertConfigureFails(StartStopCut(), {'threshold': 1})
        self.assertConfigureFails(StartStopCut(), {'maximumStartTime': -.5})
        self.assertConfigureFails(StartStopCut(), {'maximumStopTime': -.5})
        self.assertConfigureFails(StartStopCut(), {'frameSize': 64,
                                                   'hopSize': 65})

    def testStreamingRegression(self):
        # Streaming mode should also be tested to ensure it works well
        # with the real accumulator.
        import essentia.streaming as estr
        loader = estr.MonoLoader(filename=join(testdata.audio_dir,
                                 'recorded/mozart_c_major_30sec.wav'))
        realAccumulator = estr.RealAccumulator()
        startStopCut = estr.StartStopCut()
        pool = Pool()

        loader.audio >> realAccumulator.data

        realAccumulator.array >> startStopCut.audio

        startStopCut.startCut >> (pool, 'start')
        startStopCut.stopCut >> (pool, 'stop')

        essentia.run(loader)

        self.assertEqual(pool['start'], 0)
        self.assertEqual(pool['stop'], 1)

suite = allTests(TestStartStopCut)


if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
