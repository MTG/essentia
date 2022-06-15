#!/usr/bin/env python

# Copyright (C) 2006-2022  Music Technology Group - Universitat Pompeu Fabra
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
import essentia.streaming as streaming

class TestTensorFlowPredict2D(TestCase):

    # For efficiency, compute the expected results once and store them in the test class.
    def __init__(self, args):
        super().__init__(args)

        # Let's use MusiCNN for testing. If it uses half-overlapped 3-second patches and
        # we want predictions from 2 patches, we need 4.5 seconds of audio.
        self.sr = 16000
        self.time = 4.5
        self.nsamples = int(self.sr * self.time)

        # Using synthetic noise to speed up the tests.
        numpy.random.seed(6)
        self.audio = numpy.random.rand(self.nsamples).astype("float32")

        # Using `genre_dortmund_musicnn_msd.pb` because it is the MusiCNN model we have in our repo.
        # We can extract the generic MusiCNN embeddings from the first dense layer of the model.
        self.embedding_model_name = join(testdata.models_dir, 'musicnn', 'genre_dortmund_musicnn_msd.pb')
        # A typical use case of this algorithm is the inference with embedding-based classifiers.
        self.head_model_name = join(testdata.models_dir, 'classification_heads', 'emomusic-msd-musicnn-1.pb')

        self.head_input_name = "flatten_in_input"
        self.head_output_name = "dense_out"

        # Get the expected results with the generic TensorflowPredict() algorithm.
        self.embeddings, self.expected = self.getExpectedResults()

    def getExpectedResults(self):
        embedding_model = TensorflowPredictMusiCNN(
            graphFilename=self.embedding_model_name,
            output="model/dense/BiasAdd"
        )
        classification_head = TensorflowPredict(
            graphFilename=self.head_model_name,
            inputs=[self.head_input_name],
            outputs=[self.head_output_name]
        )

        embeddings = embedding_model(self.audio)
        embeddings_tensor = numpy.expand_dims(embeddings, (1, 2))
        pool = Pool()
        pool.set(self.head_input_name, embeddings_tensor)

        pool_out = classification_head(pool)
        expected = pool_out[self.head_output_name].squeeze()
        return embeddings, expected


    def testRegressionStandard(self):
        classification_head = TensorflowPredict2D(
            graphFilename=self.head_model_name,
            input=self.head_input_name,
            output=self.head_output_name,
        )

        found = classification_head(self.embeddings)

        self.assertAlmostEqualMatrix(found, self.expected)

    def testRegressionStandardCustomDimension(self):
        # In standard mode, `dimensions` should be overridden by the
        # actual input data shape.
        classification_head = TensorflowPredict2D(
            graphFilename=self.head_model_name,
            input=self.head_input_name,
            output=self.head_output_name,
            dimensions=100,
        )

        found = classification_head(self.embeddings)

        self.assertAlmostEqualMatrix(found, self.expected)


    def testRegressionStreaming(self):
        # Do end-to-end extraction in streaming mode.
        embedding_model = streaming.TensorflowPredictMusiCNN(
            graphFilename=self.embedding_model_name,
            output="model/dense/BiasAdd"
        )
        classification_head = streaming.TensorflowPredict2D(
            graphFilename=self.head_model_name,
            input=self.head_input_name,
            output=self.head_output_name,
        )
        pool = Pool()
        vectorInput = VectorInput(self.audio)

        vectorInput.data >> embedding_model.signal
        embedding_model.predictions >> classification_head.features
        classification_head.predictions >> (pool, "predictions")

        run(vectorInput)
        found = pool["predictions"]

        self.assertAlmostEqualMatrix(found, self.expected)

    def testEmptyModelName(self):
        # With empty model names the algorithm should skip the configuration without errors.
        self.assertConfigureSuccess(TensorflowPredict2D(), {})
        self.assertConfigureSuccess(TensorflowPredict2D(), {'graphFilename': ''})
        self.assertConfigureSuccess(TensorflowPredict2D(), {'graphFilename': '',
                                                            'input': '',
                                                                })
        self.assertConfigureSuccess(TensorflowPredict2D(), {'graphFilename': '',
                                                            'input': 'wrong_input'
                                                                })
        self.assertConfigureSuccess(TensorflowPredict2D(), {'savedModel': ''})
        self.assertConfigureSuccess(TensorflowPredict2D(), {'savedModel': '',
                                                            'input':'',
                                                                })
        self.assertConfigureSuccess(TensorflowPredict2D(), {'savedModel': '',
                                                            'input':'wrong_input',
                                                                })
        self.assertConfigureSuccess(TensorflowPredict2D(), {'graphFilename': '',
                                                            'savedModel':'',
                                                                })
        self.assertConfigureSuccess(TensorflowPredict2D(), {'graphFilename': '',
                                                            'savedModel':'',
                                                            'input': '',
                                                                })
        self.assertConfigureSuccess(TensorflowPredict2D(), {'graphFilename': '',
                                                            'savedModel':'',
                                                            'input': 'wrong_input',
                                                                })


    def testInvalidParam(self):
        model = join(testdata.models_dir, 'vgg', 'vgg4.pb')
        self.assertConfigureFails(TensorflowPredict2D(), {'graphFilename': model,
                                                          'input': 'wrong_head_input_name',
                                                          'output': 'model/Softmax',
                                                               })  # input do not exist in the model
        self.assertConfigureFails(TensorflowPredict2D(), {'graphFilename': 'wrong_model_name',
                                                          'input': 'model/Placeholder',
                                                          'output': 'model/Softmax',
                                                               })  # the model does not exist

suite = allTests(TestTensorFlowPredict2D)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
