#!/usr/bin/env python


#
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
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/

from essentia_test import *

class TestDCRemoval(TestCase):

    def testInvalidParam(self):
        self.assertConfigureFails(DCRemoval(), { 'sampleRate': -23 })
        self.assertConfigureFails(DCRemoval(), { 'cutoffFrequency': 0 })


    def testEmpty(self):
        self.assertEqualVector(DCRemoval()([]), [])


    def testOneByOne(self):
        # we compare here that filtering an array all at once or the samples
        # one by one will yield the same result
        filt = DCRemoval()
        signal = readVector(join(filedir(), 'filters/x.txt'))

        expected = filt(signal)

        # need to reset the filter here!!
        filt.reset()

        result = []
        for sample in signal:
            result += list(filt([sample]))

        self.assertAlmostEqualVector(result, expected)

    def testZero(self):
        self.assertEqualVector(DCRemoval()(zeros(20)), zeros(20))


    def testConstantInput(self):
        # we only test starting from the 1000th position, because we need to wait
        # for the filter to stabilize
        self.assertAlmostEqualVector(DCRemoval()(ones(20000))[1000:], zeros(20000)[1000:], 3.3e-3)

    def testRegression(self):
        signal = MonoLoader(filename = join(testdata.audio_dir,
                                            'generated', 'doublesize',
                                            'sin_30_seconds.wav'),
                            sampleRate = 44100)()

        dcOffset = 0.2
        dcsignal = signal + dcOffset

        self.assertAlmostEqual(numpy.mean(DCRemoval()(dcsignal)[2500:]),0, 1e-6)


suite = allTests(TestDCRemoval)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
