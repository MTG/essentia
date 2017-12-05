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
from essentia.standard import AudioLoader, StereoMuxer

class TestStereoMuxer(TestCase):

    def testRegression(self):
        size = 10
        result = StereoMuxer()([size-i for i in range(size)], [i for i in range(size)])
        expected = array([[size-i, i] for i in range(size)])

        result_l, result_r = zip(*result)
        expected_l, expected_r = zip(*expected)
        self.assertEqualVector(result_l, expected_l)
        self.assertEqualVector(result_r, expected_r)

    def testEmpty(self):
        audio = StereoMuxer()([], [])
        self.assertEqualVector(audio, [])


suite = allTests(TestStereoMuxer)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
