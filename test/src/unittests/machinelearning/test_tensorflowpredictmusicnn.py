#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
import os


class TestTensorFlowPredictMusiCNN(TestCase):

    def testRegression(self):
        expected = [0.00893995, 0.650628, 0.0017924, 0.10297492, 0.15139213,
                    0.13975027, 0.07960042, 0.01067388, 0.04935676]

        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        model = join(testdata.models_dir, 'musicnn', 'genre_dortmund_musicnn_msd.pb')

        audio = MonoLoader(filename=filename, sampleRate=16000)()
        found = TensorflowPredictMusiCNN(graphFilename=model)(audio)[0]

        self.assertAlmostEqualVector(found, expected, 1e-6)


suite = allTests(TestTensorFlowPredictMusiCNN)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
