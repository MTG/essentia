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

    # def testInference(self,):
    #     model = join(testdata.models_dir, "effnetdiscogs", "effnetdiscogs-bsdynamic-1.onnx")

    #     # define input and output metadata
    #     input_shape = (1, 128, 96)
    #     outputs = [
    #         {
    #             "name": "activations",
    #             "shape": (1, 400),
    #         },
    #         {
    #             "name": "embeddings",
    #             "shape": (1, 1280),
    #         }
    #     ]

    #     onxx_predict = OnnxPredict(
    #         graphFilename= model,
    #         inputs=[],
    #         outputs=[output["name"] for output in outputs],
    #         )

    #     stem = "359500__mtg__sax-tenor-e-major"
    #     audio_path = join(testdata.audio_dir, Path("recorded"), f"{stem}.wav")

    #     audio, sample_rate = sf.read(audio_path, dtype=numpy.float32)

    #     frame_size = 512
    #     hop_size = 256
    #     patch_size = 128
    #     number_bands = 96

    #     w = Windowing(type="hann", zeroPadding=frame_size)
    #     spectrum = Spectrum(size=frame_size)
    #     mels = MelBands(inputSize=frame_size+1,numberBands=number_bands, type="magnitude")
    #     logNorm = UnaryOperator(type="log")

    #     # compute mel bands
    #     bands = []
    #     for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
    #         melFrame = mels(spectrum(w(frame)))
    #         bands.append(logNorm(melFrame))
    #     bands = array(bands)

    #     discard = bands.shape[0] % patch_size
    #     bands = numpy.reshape(bands[:-discard, :], [-1, patch_size, number_bands])
    #     batch = numpy.expand_dims(bands, 1)

    #     pool = Pool()
    #     pool.set("melspectrogram", batch)

    #     pool_out = onxx_predict(pool)

    #     self.assertEqualVector(input_shape, batch.shape[1:])
    #     self.assertEqualVector(outputs[0]["shape"], pool_out[outputs[0]["name"]].shape[2:])
    #     self.assertEqualVector(outputs[1]["shape"], pool_out[outputs[1]["name"]].shape[2:])
    #     self.assertEqual(pool_out.descriptorNames()[0], outputs[0]["name"])
    #     self.assertEqual(pool_out.descriptorNames()[1], outputs[1]["name"])

    def testInference(self,):

        # define output metadata
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

        onxx_predict = OnnxPredict()
        pool = Pool()

        pool_out = runEffnetDiscogsInference(onxx_predict, outputs, pool)

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

        # prepare model inputs and batches
        input1, input2 = (numpy.float32(numpy.random.random((3, 3))) for _ in range(2))

        n, m = input1.shape

        batch1 = input1.reshape(n, 1, 1, m)
        batch2 = input2.reshape(n, 1, 1, m)

        pool = Pool()
        pool.set("input1", batch1)
        pool.set("input2", batch2)

        found_values1, found_values2 = runIdentityModelInference(OnnxPredict(), pool)

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
    def testConfiguration(self):
        # define output metadata
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

        onxx_predict = OnnxPredict()
        pool = Pool()

        _ = runEffnetDiscogsInference(onxx_predict, outputs, pool)
        pool.clear()

        # prepare model inputs and batches for identity model
        input1, input2 = (numpy.float32(numpy.random.random((3, 3))) for _ in range(2))

        n, m = input1.shape

        batch1 = input1.reshape(n, 1, 1, m)
        batch2 = input2.reshape(n, 1, 1, m)

        pool.set("input1", batch1)
        pool.set("input2", batch2)

        found_values1, found_values2 = runIdentityModelInference(onxx_predict, pool)

        self.assertAlmostEqualMatrix(found_values1, batch1)
        self.assertAlmostEqualMatrix(found_values2, batch2)

def runEffnetDiscogsInference(onnx_predict, outputs, pool) -> Pool:
    model = join(testdata.models_dir, "effnetdiscogs", "effnetdiscogs-bsdynamic-1.onnx")

    stem = "359500__mtg__sax-tenor-e-major"
    audio_path = join(testdata.audio_dir, Path("recorded"), f"{stem}.wav")

    audio, _ = sf.read(audio_path, dtype=numpy.float32)

    onnx_predict.configure(
        graphFilename= model,
        inputs=[],
        outputs=[output["name"] for output in outputs],
        )

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

    pool.set("melspectrogram", batch)

    return onnx_predict(pool)

def runIdentityModelInference(onnx_predict, pool):
    model = join(testdata.models_dir, "identity", "identity2x2.onnx")

    onnx_predict.configure(
        graphFilename=model,
        inputs=["input1", "input2"],
        outputs=["output1", "output2"],
        squeeze=True,
    )

    poolOut = onnx_predict(pool)

    return poolOut["output1"], poolOut["output2"]


suite = allTests(TestOnnxPredict)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
