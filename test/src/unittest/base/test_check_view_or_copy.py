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
import sys
import numpy as np

class TestCheckViewOrCopy(TestCase):
    def testCreatesNewCopy(self):
        mono = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'))()
        audio = AudioLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'))()[0]
        w = Windowing()

        # This case is known to produce different values if it the data is not copied to a new array.  
        mono_frame = mono[0:1024]
        audio_frame = audio[0:1024,0]

        # Here we test that the python parser detects that inputs are views and creates a copy of them.
        self.assertEqualVector(w(mono_frame), w(audio_frame))

suite = allTests(TestCheckViewOrCopy)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)