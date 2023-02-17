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


class TestTensorFlowPredictEffnetDiscogs(TestCase):

    def regression(self, parameters):
        expected = numpy.load(
            join(filedir(), "tensorflowpredicteffnetdiscogs", "effnetdiscogs.activations.npy")
        )

        filename = join(testdata.audio_dir, 'recorded', 'techno_loop.wav')
        audio = MonoLoader(filename=filename, sampleRate=16000)()
        
        activations = TensorflowPredictEffnetDiscogs(**parameters)(audio)
        found = numpy.mean(activations, axis=0)

        self.assertAlmostEqualVector(found, expected, 1e-4)

    def testRegressionSavedModel(self):
        parameters = {
            "savedModel": join(testdata.models_dir, 'effnetdiscogs', 'effnetdiscogs-bs64-1'),
        }
        self.regression(parameters)

    def testRegressionFrozenModel(self):
        parameters = {
            "graphFilename": join(testdata.models_dir, 'effnetdiscogs', 'effnetdiscogs-bs64-1.pb'),
        }
        self.regression(parameters)

    def testLastBatchMode(self):
        sr = 16000
        batch_size = 64
        for last_batch_mode in ("discard", "zeros", "same"):
            model = TensorflowPredictEffnetDiscogs(
                graphFilename=join(testdata.models_dir, 'effnetdiscogs', 'effnetdiscogs-bs64-1.pb'),
                lastBatchMode=last_batch_mode,
                batchSize=batch_size,
            )

            # It'd be nice to test other durations but we are limited by inference times,
            # especially in CPU.
            seconds = 150
            audio = numpy.ones((sr * seconds), dtype="float32")
            found = len(model(audio))

            if last_batch_mode == "discard":
                expected = (seconds // batch_size) * batch_size
            elif last_batch_mode == "zeros":
                expected = numpy.ceil(seconds / batch_size) * batch_size
            elif last_batch_mode == "same":
                expected = seconds

            self.assertEqual(found, expected)

    def testInvalidParam(self):
        model = join(testdata.models_dir, 'effnetdiscogs', 'effnetdiscogs-bs64-1.pb')
        self.assertConfigureFails(TensorflowPredictEffnetDiscogs(), {'graphFilename': model,
                                                                     'batchSize': -2,
                                                                    })  # Cannot be < -1.
        self.assertConfigureFails(TensorflowPredictEffnetDiscogs(), {'graphFilename': model,
                                                                     'patchSize': 0,
                                                                    })  # Cannot be 0.

suite = allTests(TestTensorFlowPredictEffnetDiscogs)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
