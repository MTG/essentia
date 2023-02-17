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


class TestTensorFlowPredictVGGish(TestCase):

    def testRegressionFrozenModel(self):
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

    def testPatchSize(self):
        model = join(testdata.models_dir, 'vggish', 'small_vggish_init.pb')
        # One second of audio.
        data = numpy.ones((16000)).astype("float32")

        # VGGish expects a fixed batch size of 96 frames.
        patch_size = 96
        TensorflowPredictVGGish(
            graphFilename=model,
            patchHopSize=0,
            patchSize=patch_size,
        )(data)

        # An exception should be risen otherwise.
        patch_size = 65
        self.assertComputeFails(
            TensorflowPredictVGGish(
                graphFilename=model,
                patchHopSize=0,
                patchSize=patch_size,
            ),
            data
        )

    def testRegressionSavedModel(self):
        expected = [0.49044114, 0.50241125]

        filename = join(testdata.audio_dir, 'recorded', 'hiphop.mp3')
        model = join(testdata.models_dir, 'vggish', 'small_vggish_init')

        audio = MonoLoader(filename=filename, sampleRate=16000)()
        found = TensorflowPredictVGGish(savedModel=model, patchHopSize=0)(audio)
        found = numpy.mean(found, axis=0)

        # Setting a high tolerance value due to the mismatch between the
        # original and replicated features. However, they are close enough to
        # make valid predictions.
        self.assertAlmostEqualVector(found, expected, 1e-2)

    def testEmptyModelName(self):
        # With empty model names the algorithm should skip the configuration without errors.
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {})
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'graphFilename': ''})
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'graphFilename': '',
                                                                'input': '',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'graphFilename': '',
                                                                'input': 'wrong_input'
                                                               })
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'savedModel': ''})
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'savedModel': '',
                                                                'input':'',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'savedModel': '',
                                                                'input':'wrong_input',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'graphFilename': '',
                                                                'savedModel':'',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'graphFilename': '',
                                                                'savedModel':'',
                                                                'input': '',
                                                               })
        self.assertConfigureSuccess(TensorflowPredictVGGish(), {'graphFilename': '',
                                                                'savedModel':'',
                                                                'input': 'wrong_input',
                                                               })
    
    def testInvalidParam(self):
        model = join(testdata.models_dir, 'vgg', 'vgg4.pb')
        self.assertConfigureFails(TensorflowPredictVGGish(), {'graphFilename': model,
                                                              'input': 'wrong_input_name',
                                                              'output': 'model/Softmax',
                                                             })  # input does not exist in the model
        self.assertConfigureFails(TensorflowPredictVGGish(), {'graphFilename': 'wrong_model_name',
                                                              'input': 'model/Placeholder',
                                                              'output': 'model/Softmax',
                                                             })  # the model does not exist


suite = allTests(TestTensorFlowPredictVGGish)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
