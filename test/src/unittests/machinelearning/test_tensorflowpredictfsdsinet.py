#!/usr/bin/env python

# Copyright (C) 2006-2023  Music Technology Group - Universitat Pompeu Fabra
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


class TestTensorFlowPredictFSDSINet(TestCase):
    sr = 22050

    def testRegressionDC(self):
        """Regression of the activations for a DC signal"""

        expected_file = join(
            filedir(),
            'tensorflowpredictfsdsinet',
            'dc_activations_fsd-sinet-vgg41-tlpf-ibp-1.npy'
        )
        expected = numpy.load(expected_file)

        audio = numpy.ones(self.sr, dtype="float32")

        model = join(
            testdata.models_dir,
            'fsd-sinet',
            'fsd-sinet-vgg41-tlpf-ibp-1.pb'
        )
        found = TensorflowPredictFSDSINet(graphFilename=model, normalize=False)(audio).squeeze()

        self.assertAlmostEqualVector(found, expected, 1e-3)

    def testRegressionVoiceSignal(self):
        """Regression of the activations for a voice signal"""

        expected_file = join(
            filedir(),
            'tensorflowpredictfsdsinet',
            'vignesh_activations_fsd-sinet-vgg41-tlpf-ibp-1.npy'
        )
        expected = numpy.load(expected_file)

        audio_file = join(testdata.audio_dir, 'recorded/vignesh.wav')
        audio = MonoLoader(filename=audio_file, sampleRate=self.sr)()

        segment = audio[:self.sr]

        model = join(
            testdata.models_dir,
            'fsd-sinet',
            'fsd-sinet-vgg41-tlpf-ibp-1.pb'
        )
        found = TensorflowPredictFSDSINet(graphFilename=model)(segment).squeeze()

        self.assertAlmostEqualVector(found, expected, 1e-3)

    def testRegressionFrozenModelFromSpecPatch(self):
        """ Regression of the activations from a pre-computed mel-spectrogram patch

        This test skips the mel-spectrogram computation part to focus on possible
        problems with the tensorflow backend.
        """

        model = join(
            testdata.models_dir,
            'fsd-sinet',
            'fsd-sinet-vgg41-tlpf-ibp-1.pb'
        )
        melspectrogram_file = join(
            filedir(),
            'tensorflowpredictfsdsinet',
            'patch_id1013_t55_156_f96_waterdrip_mel_spectrogram.npy'
        )
        expected_file = join(
            filedir(),
            'tensorflowpredictfsdsinet',
            'patch_id1013_t55_156_f96_waterdrip_scores_TLPF5x5_IBP_vgg41.npy'
        )

        mel_spectrogram = numpy.load(melspectrogram_file).astype("float32")

        mel_spectrogram = numpy.expand_dims(mel_spectrogram, 0)
        mel_spectrogram = numpy.expand_dims(mel_spectrogram, -1)

        expected = numpy.load(expected_file)

        pool = Pool()
        input_name = "x"
        output_name = "model/predictions/Sigmoid"
        pool.set(input_name, mel_spectrogram)

        predictor = TensorflowPredict(
            graphFilename=model,
            inputs=[input_name],
            outputs=[output_name],
            squeeze=False,
        )

        found = predictor(pool)[output_name]
        found = found.squeeze()

        self.assertAlmostEqualVector(found, expected, 1e-3)

    def testEmptyModelName(self):
        """Test empty model names"""

        # With empty model names the algorithm should skip the configuration without errors.
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {})
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'graphFilename': ''})
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'graphFilename': '',
                                                                 'input': '',
                                                                })
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'graphFilename': '',
                                                                 'input': 'wrong_input'
                                                                })
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'savedModel': ''})
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'savedModel': '',
                                                                 'input':'',
                                                                })
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'savedModel': '',
                                                                 'input':'wrong_input',
                                                                })
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'graphFilename': '',
                                                                 'savedModel':'',
                                                                })
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'graphFilename': '',
                                                                 'savedModel':'',
                                                                 'input': '',
                                                                })
        self.assertConfigureSuccess(TensorflowPredictFSDSINet(), {'graphFilename': '',
                                                                 'savedModel':'',
                                                                 'input': 'wrong_input',
                                                                })


    def testInvalidParam(self):
        """Test invalid parameters"""

        model = join(testdata.models_dir, 'vgg', 'vgg4.pb')
        self.assertConfigureFails(TensorflowPredictFSDSINet(), {'graphFilename': model,
                                                               'input': 'wrong_input_name',
                                                               'output': 'model/Softmax',
                                                               })  # input do not exist in the model
        self.assertConfigureFails(TensorflowPredictFSDSINet(), {'graphFilename': 'wrong_model_name',
                                                               'input': 'model/Placeholder',
                                                               'output': 'model/Softmax',
                                                               })  # the model does not exist

suite = allTests(TestTensorFlowPredictFSDSINet)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
