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

class TestTensorflowPredictCREPE(TestCase):

    def regression(self, parameters):
        expected_activations = numpy.load(
            join(filedir(),
            'tensorflowpredict_crepe',
            'vignesh.activation.npy')
        )

        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        model = join(testdata.models_dir, 'crepe', 'crepe-tiny-1.pb')

        audio = MonoLoader(filename=filename, sampleRate=16000)()
        activations = TensorflowPredictCREPE(**parameters)(audio)

        self.assertAlmostEqualMatrix(activations, expected_activations, 1e-2)


    def testRegressionGraphFilename(self):
        self.regression({'graphFilename': join(testdata.models_dir, 'crepe', 'crepe-tiny-1.pb')})

    def testRegressionSavedModel(self):
        self.regression({'savedModel': join(testdata.models_dir, 'crepe', 'crepe-tiny-1')})


    def testEmptyModelName(self):
        # With empty or undefined model names the algorithm should skip the configuration
        # without errors.
        TensorflowPredictCREPE()
        TensorflowPredictCREPE(graphFilename='')
        TensorflowPredictCREPE(graphFilename='', input='')
        TensorflowPredictCREPE(graphFilename='', input='wrong_input')

    def testInvalidParam(self):
        model = join(testdata.models_dir, 'crepe', 'crepe-tiny-1.pb')
        self.assertConfigureFails(TensorflowPredictCREPE(), {'graphFilename': model,
                                                             'input': 'wrong_input_name',
                                                             'output': 'model/classifier/Sigmoid',
                                                            })  # input do not exist in the model
        self.assertConfigureFails(TensorflowPredictCREPE(), {'graphFilename': 'wrong_model_name',
                                                             'input': 'frames',
                                                             'output': 'model/classifier/Sigmoid',
                                                            })  # the model does not exist

suite = allTests(TestTensorflowPredictCREPE)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
