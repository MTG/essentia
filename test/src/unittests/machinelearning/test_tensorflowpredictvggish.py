#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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


class TestTensorFlowPredictVGGish(TestCase):

    def testRegression(self):
        expected = [0.49044114, 0.50241125]

        filename = join(testdata.audio_dir, 'recorded', 'hiphop.mp3')
        model = join(testdata.models_dir, 'vggish', 'small_vggish_init.pb')

        audio = MonoLoader(filename=filename, sampleRate=16000)()
        found = TensorflowPredictVGGish(graphFilename=model, patchHopSize=0)(audio)
        found = numpy.mean(found, axis=0)

        # Setting a high tolerance value due to the mismatch between the
        # original and replicated features. However, they are close enough to
        # make valid predictions.
        self.assertAlmostEqualVector(found, expected, 1e-2)


suite = allTests(TestTensorFlowPredictVGGish)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
