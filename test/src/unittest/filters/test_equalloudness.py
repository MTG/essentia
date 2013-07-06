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

class TestEqualLoudness(TestCase):


    def testInvalidParam(self):
        self.assertConfigureFails(EqualLoudness(), { 'sampleRate': -23 })


    def testEmpty(self):
        self.assertEqualVector(EqualLoudness()([]), [])


    def testOneByOne(self):
        # we compare here that filtering an array all at once or the samples
        # one by one will yield the same result
        filt = EqualLoudness()
        signal = readVector(join(filedir(), 'filters/x.txt'))

        expected = filt(signal)

        # need to reset the filter here!!
        filt.reset()

        result = []
        for sample in signal:
            result += list(filt([sample]))

        self.assertAlmostEqualVector(result, expected, 1e-3)

    def testZero(self):
        self.assertEqualVector(EqualLoudness()(zeros(20)), zeros(20))

    def testRegression(self):
        signal = MonoLoader(filename = join(testdata.audio_dir, 'generated', 'doublesize', 'sin_30_seconds.wav'),
                            sampleRate = 44100)()[:100000]
        expected = MonoLoader(filename = join(testdata.audio_dir, 'generated', 'doublesize', 'sin_30_seconds_eqloud.wav'),
                              sampleRate = 44100)()[:100000]

        # assert on the difference of the signals here, because we want the absolute
        # difference, not a relative one
        self.assertAlmostEqualVector(EqualLoudness()(signal) - expected, zeros(len(expected)), 1e-4)


suite = allTests(TestEqualLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
