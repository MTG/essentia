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
from pathlib import Path
import soundfile as sf


class TestOnnxPredict(TestCase):
    # def testIONameParser(self):
    #     model = join(testdata.models_dir, "effnetdiscogs", "effnetdiscogs-bsdynamic-1.onnx")
    #     print(f"\nmodel: {model}")
    #     configs = [
    #         {
    #             "graphFilename": model,
    #             "inputs": ["model/Placeholder"],
    #             "outputs": ["model/Softmax:"],
    #         },  # No index.
    #         {
    #             "graphFilename": model,
    #             "inputs": ["model/Placeholder"],
    #             "outputs": ["model/Softmax:3"],
    #         },  # Index out of bounds.
    #         {
    #             "graphFilename": model,
    #             "inputs": ["model/Placeholder"],
    #             "outputs": ["model/Softmax::0"],
    #         },  # Double colon.
    #         {
    #             "graphFilename": model,
    #             "inputs": ["model/Placeholder"],
    #             "outputs": ["model/Softmax:s:0"],
    #         },  # Several colons.
    #     ]

    #     for config in configs[1:]:
    #         with self.subTest(f"{config} failed"):
    #             print(config)
    #             self.assertConfigureFails(OnnxPredict(), config)

    def testInference(self,):
        model = join(testdata.models_dir, "effnetdiscogs", "effnetdiscogs-bsdynamic-1.onnx")

        # define input and output groundtruths
        input_shape = (1, 128, 96)
        outputs = [
            {
                "name": "activations",
                "shape": (1, 400),
            },
            {
                "name": "embeddings",
                "shape": (1, 1280),
            }
        ]

        onxx_predict = OnnxPredict(
            graphFilename= model,
            inputs=[],
            outputs=[output["name"] for output in outputs],
            )

        stem = "359500__mtg__sax-tenor-e-major"
        audio_path = join(testdata.audio_dir, Path("recorded"), f"{stem}.wav")

        audio, sample_rate = sf.read(audio_path, dtype=numpy.float32)

        frame_size = 512
        hop_size = 256
        patch_size = 128
        number_bands = 96

        w = Windowing(type="hann", zeroPadding=frame_size)
        spectrum = Spectrum(size=frame_size)
        mels = MelBands(inputSize=frame_size+1,numberBands=number_bands, type="magnitude")
        logNorm = UnaryOperator(type="log")

        # compute mel bands
        bands = []
        for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
            melFrame = mels(spectrum(w(frame)))
            bands.append(logNorm(melFrame))
        bands = array(bands)

        discard = bands.shape[0] % patch_size
        bands = numpy.reshape(bands[:-discard, :], [-1, patch_size, number_bands])
        batch = numpy.expand_dims(bands, 1)

        pool = Pool()
        pool.set("melspectrogram", batch)

        pool_out = onxx_predict(pool)

        self.assertEqualVector(input_shape, batch.shape[1:])
        self.assertEqualVector(outputs[0]["shape"], pool_out[outputs[0]["name"]].shape[2:])
        self.assertEqualVector(outputs[1]["shape"], pool_out[outputs[1]["name"]].shape[2:])
        self.assertEqual(pool_out.descriptorNames()[0], outputs[0]["name"])
        self.assertEqual(pool_out.descriptorNames()[1], outputs[1]["name"])

    def testEmptyModelName(self):
        # With empty model name the algorithm should skip the configuration without errors.
        self.assertConfigureSuccess(OnnxPredict(), {})
        self.assertConfigureSuccess(OnnxPredict(), {"graphFilename": ""})
        self.assertConfigureSuccess(
            OnnxPredict(), {"graphFilename": "", "inputs": [""]}
        )
        self.assertConfigureSuccess(
            OnnxPredict(), {"graphFilename": "", "inputs": ["wrong_input"]}
        )

    def testInvalidParam(self):
        model = join(testdata.models_dir, "effnetdiscogs", "effnetdiscogs-bsdynamic-1.onnx")
        self.assertConfigureFails(
            OnnxPredict(),
            {
                "graphFilename": model,
                "inputs": ["wrong_input_name"],
                "outputs": ["embeddings"],
            },
        )  # input does not exist in the model
        self.assertConfigureFails(
            OnnxPredict(),
            {
                "graphFilename": "wrong_model_name",    #! I suspect the issue is here with OnnxExceptions
                "inputs": ["melspectrogram"],
                "outputs": ["embeddings"],
            },
        )  # the model does not exist

    def testIdentityModel(self):
        model = join(testdata.models_dir, "identity", "identity2x2.onnx")

        # prepare model inputs and batches
        input1, input2 = (numpy.float32(numpy.random.random((3, 3))) for _ in range(2))

        n, m = input1.shape

        batch1 = input1.reshape(n, 1, 1, m)
        batch2 = input2.reshape(n, 1, 1, m)

        pool = Pool()
        pool.set("input1", batch1)
        pool.set("input2", batch2)

        poolOut = OnnxPredict(
            graphFilename=model,
            inputs=["input1", "input2"],
            outputs=["output1", "output2"],
            squeeze=True,
        )(pool)

        found_values1 = poolOut["output1"]
        found_values2 = poolOut["output2"]

        self.assertAlmostEqualMatrix(found_values1, batch1)
        self.assertAlmostEqualMatrix(found_values2, batch2)

    def testComputeWithoutConfiguration(self):
        pool = Pool()
        pool.set("melspectrogram", numpy.zeros((1, 1, 1, 1), dtype="float32"))

        self.assertComputeFails(OnnxPredict(), pool)

    def testIgnoreInvalidReconfiguration(self):
        pool = Pool()
        pool.set("input1", numpy.ones((1, 1, 1, 3), dtype="float32"))
        pool.set("input2", numpy.ones((1, 1, 1, 3), dtype="float32"))

        #model_name = join(filedir(), "tensorflowpredict", "identity.pb")
        model_name = join(testdata.models_dir, "identity", "identity2x2.onnx")
        model = OnnxPredict(
            graphFilename=model_name,
            inputs=["input1", "input2"],
            outputs=["output1"],
            squeeze=True,
        )

        firstResult = model(pool)

        # This attempt to reconfigure the algorithm should be ignored and trigger a Warning.
        model.configure()

        secondResult = model(pool)

        self.assertEqualMatrix(firstResult["output1"], secondResult["output1"])

    # TODO: make a test for squeeze, showing that it works well when it is applied for a 2D model
    def testInvalidSqueezeConfiguration(self):
        model = join(testdata.models_dir, "identity", "identity2x2.onnx")

        # prepare model inputs and batches
        input1, input2 = (numpy.float32(numpy.random.random((3, 3))) for _ in range(2))

        n, m = input1.shape

        batch1 = input1.reshape(n, 1, 1, m)
        batch2 = input2.reshape(n, 1, 1, m)

        pool = Pool()
        pool.set("input1", batch1)
        pool.set("input2", batch2)

        onnx_predict = OnnxPredict(
            graphFilename=model,
            inputs=["input1", "input2"],
            outputs=["output1", "output2"],
            squeeze=False,
        )
        self.assertComputeFails(onnx_predict, pool)

    # TODO: make a test reusing the algorithm for two models (effnet and identity)

    """
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
    """

suite = allTests(TestOnnxPredict)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
