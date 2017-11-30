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


class TestCheckViewOrCopy(TestCase):
    def testComputeNewCopy(self):
        mono = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'))()
        audio = AudioLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'))()[0]
        w = Windowing()

        # This case is known to produce different values if it the data is not copied to a new array.
        mono_frame = mono[0:1024]
        audio_frame = audio[0:1024, 0]

        # Here we test that the python parser detects that inputs are views and creates a copy of them.
        self.assertEqualVector(w(mono_frame), w(audio_frame))

    def testConfigureNewCopy(self):

        mono = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'))()
        mono_frame = mono[0:1024]
        spectrum = Spectrum()(Windowing()(mono_frame))

        bands = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        bands1 = [500, 600, 700, 800, 900, 1000]
        bands2 = bands[5:]

        # Here we test that the python parser detects that parameters are views and creates a copy of them.
        fbands1 = FrequencyBands(frequencyBands=bands1)
        fbands2 = FrequencyBands(frequencyBands=bands2)

        self.assertEqualVector(fbands1(spectrum), fbands2(spectrum))


suite = allTests(TestCheckViewOrCopy)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
