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


class TestTensorFlowPredict(TestCase):
    def regression(self, parameters):
        # Test a simple tensorflow model trained on Essentia features.
        # The ground true values were obtained with the following script:
        expectedValues = [9.9997020e-01, 5.3647455e-14, 2.9801236e-05, 4.9495230e-12]

        patchSize = 128
        numberBands = 128
        frameSize = 1024
        hopSize = frameSize

        filename = join(testdata.audio_dir, "recorded", "cat_purrrr.wav")

        audio = MonoLoader(filename=filename)()

        w = Windowing(type="hann", zeroPadding=frameSize)
        spectrum = Spectrum()
        mels = MelBands(numberBands=numberBands, type="magnitude")
        logNorm = UnaryOperator(type="log")

        bands = []
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            melFrame = mels(spectrum(w(frame)))
            bands.append(logNorm(melFrame))
        bands = array(bands)

        discard = bands.shape[0] % patchSize
        bands = numpy.reshape(bands[:-discard, :], [-1, patchSize, numberBands])
        batch = numpy.expand_dims(bands, 1)

        pool = Pool()
        pool.set("model/Placeholder", batch)

        tfp = TensorflowPredict(**parameters)
        poolOut = tfp(pool)

        foundValues = poolOut["model/Softmax"].mean(axis=0).squeeze()

        self.assertAlmostEqualVector(foundValues, expectedValues, 1e-5)

    def testRegressionFrozenModel(self):
        parameters = {
            "graphFilename": join(testdata.models_dir, "vgg", "vgg4.pb"),
            "inputs": ["model/Placeholder"],
            "outputs": ["model/Softmax"],
            "isTraining": False,
            "isTrainingName": "model/Placeholder_1",
        }

        self.regression(parameters)

    def testRegressionSavedModel(self):
        parameters = {
            "savedModel": join(testdata.models_dir, "vgg", "vgg4"),
            "inputs": ["model/Placeholder"],
            "outputs": ["model/Softmax"],
            "isTraining": False,
            "isTrainingName": "model/Placeholder_1",
        }

        self.regression(parameters)

    def testSavedModelOverridesGraphFilename(self):
        # When both are specified, `savedModel` should be preferred.
        # Test this by setting an invalid `graphFilename` that should be ignored.
        parameters = {
            "graphFilename": "wrong_model",
            "savedModel": join(testdata.models_dir, "vgg", "vgg4"),
            "inputs": ["model/Placeholder"],
            "outputs": ["model/Softmax"],
            "isTraining": False,
            "isTrainingName": "model/Placeholder_1",
        }

        self.regression(parameters)

    def testEmptyModelName(self):
        # With empty model name the algorithm should skip the configuration without errors.
        self.assertConfigureSuccess(TensorflowPredict(), {})
        self.assertConfigureSuccess(TensorflowPredict(), {"graphFilename": ""})
        self.assertConfigureSuccess(
            TensorflowPredict(), {"graphFilename": "", "inputs": [""]}
        )
        self.assertConfigureSuccess(
            TensorflowPredict(), {"graphFilename": "", "inputs": ["wrong_input"]}
        )
        self.assertConfigureSuccess(TensorflowPredict(), {"savedModel": ""})
        self.assertConfigureSuccess(
            TensorflowPredict(), {"savedModel": "", "inputs": [""]}
        )
        self.assertConfigureSuccess(
            TensorflowPredict(), {"savedModel": "", "inputs": ["wrong_input"]}
        )
        self.assertConfigureSuccess(
            TensorflowPredict(), {"graphFilename": "", "savedModel": ""}
        )
        self.assertConfigureSuccess(
            TensorflowPredict(), {"graphFilename": "", "savedModel": "", "inputs": [""]}
        )
        self.assertConfigureSuccess(
            TensorflowPredict(),
            {"graphFilename": "", "savedModel": "", "inputs": ["wrong_input"]},
        )

    def testInvalidParam(self):
        model = join(testdata.models_dir, "vgg", "vgg4.pb")
        self.assertConfigureFails(
            TensorflowPredict(), {"graphFilename": model}
        )  # inputs and outputs are not defined
        self.assertConfigureFails(
            TensorflowPredict(),
            {
                "graphFilename": model,
                "inputs": ["model/Placeholder"],
            },
        )  # outputs are not defined
        self.assertConfigureFails(
            TensorflowPredict(),
            {
                "graphFilename": model,
                "inputs": ["wrong_input_name"],
                "outputs": ["model/Softmax"],
            },
        )  # input does not exist in the model
        self.assertConfigureFails(
            TensorflowPredict(),
            {
                "graphFilename": "wrong_model_name",
                "inputs": ["model/Placeholder"],
                "outputs": ["model/Softmax"],
            },
        )  # the model does not exist

        # Repeat tests for savedModel format.
        model = join(testdata.models_dir, "vgg", "vgg4/")
        self.assertConfigureFails(
            TensorflowPredict(), {"savedModel": model}
        )  # inputs and outputs are not defined
        self.assertConfigureFails(
            TensorflowPredict(),
            {
                "savedModel": model,
                "inputs": ["model/Placeholder"],
            },
        )  # outputs are not defined
        self.assertConfigureFails(
            TensorflowPredict(),
            {
                "savedModel": model,
                "inputs": ["wrong_input_name"],
                "outputs": ["model/Softmax"],
            },
        )  # input does not exist in the model
        self.assertConfigureFails(
            TensorflowPredict(),
            {
                "savedModel": "wrong_model_name",
                "inputs": ["model/Placeholder"],
                "outputs": ["model/Softmax"],
            },
        )  # the model does not exist

    def testIdentityModel(self):
        # Perform the identity operation in Tensorflow to test if the data is
        # being copied correctly backwards and fordwards.
        model = join(filedir(), "tensorflowpredict", "identity.pb")
        filename = join(testdata.audio_dir, "recorded", "cat_purrrr.wav")

        audio = MonoLoader(filename=filename)()
        frames = array([frame for frame in FrameGenerator(audio)])
        batch = frames[numpy.newaxis, numpy.newaxis, :]

        pool = Pool()
        pool.set("model/Placeholder", batch)

        poolOut = TensorflowPredict(
            graphFilename=model,
            inputs=["model/Placeholder"],
            outputs=["model/Identity"],
        )(pool)

        foundValues = poolOut["model/Identity"]

        self.assertAlmostEqualMatrix(foundValues, batch)

    def testComputeWithoutConfiguration(self):
        pool = Pool()
        pool.set("model/Placeholder", numpy.zeros((1, 1, 1, 1), dtype="float32"))

        self.assertComputeFails(TensorflowPredict(), pool)

    def testIgnoreInvalidReconfiguration(self):
        pool = Pool()
        pool.set("model/Placeholder", numpy.ones((1, 1, 1, 1), dtype="float32"))

        model_name = join(filedir(), "tensorflowpredict", "identity.pb")
        model = TensorflowPredict(
            graphFilename=model_name,
            inputs=["model/Placeholder"],
            outputs=["model/Identity"],
            squeeze=False,
        )

        firstResult = model(pool)["model/Identity"]

        # This attempt to reconfigure the algorithm should be ignored and trigger a Warning.
        model.configure()

        secondResult = model(pool)["model/Identity"]

        self.assertEqualMatrix(firstResult, secondResult)

    def testImplicitOutputTensorIndex(self):
        model = join(filedir(), "tensorflowpredict", "identity.pb")
        batch = numpy.reshape(numpy.arange(4, dtype="float32"), (1, 1, 2, 2))

        pool = Pool()
        pool.set("model/Placeholder", batch)

        implicit_output = "model/Identity"
        implicit = TensorflowPredict(
            graphFilename=model,
            inputs=["model/Placeholder"],
            outputs=[implicit_output],
        )(pool)[implicit_output].squeeze()

        explicit_output = "model/Identity:0"
        explicit = TensorflowPredict(
            graphFilename=model,
            inputs=["model/Placeholder"],
            outputs=[explicit_output],
        )(pool)[explicit_output].squeeze()

        self.assertAlmostEqualMatrix(implicit, explicit)

    def testNodeNameParser(self):
        model = join(testdata.models_dir, "vgg", "vgg4.pb")

        configs = [
            {
                "graphFilename": model,
                "inputs": ["model/Placeholder"],
                "outputs": ["model/Softmax:"],
            },  # No index.
            {
                "graphFilename": model,
                "inputs": ["model/Placeholder"],
                "outputs": ["model/Softmax:3"],
            },  # Index out of bounds.
            {
                "graphFilename": model,
                "inputs": ["model/Placeholder"],
                "outputs": ["model/Softmax::0"],
            },  # Double colon.
            {
                "graphFilename": model,
                "inputs": ["model/Placeholder"],
                "outputs": ["model/Softmax:s:0"],
            },  # Several colons.
        ]

        for config in configs[1:]:
            with self.subTest(f"{config} failed"):
                self.assertConfigureFails(TensorflowPredict(), config)


suite = allTests(TestTensorFlowPredict)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
