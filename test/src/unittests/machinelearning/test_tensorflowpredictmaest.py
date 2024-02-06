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


class TestTensorFlowPredictMAEST(TestCase):
    @classmethod
    def setUpClass(self):
        # Since loading the Transformers takes a lot of time, we do it only once as reusable class members.
        # When using these algos, Essentia complains that other networks (e.g., MonoLoader's network) were
        # destroyed in the meantime. These warnings are not relevant for the tests and difficult readability,
        # so we disable them temporally.

        essentia.log.warningActive = False

        self.graphFilename30s = join(
            testdata.models_dir, "maest", "discogs-maest-30s-pw-1.pb"
        )
        self.graphFilename10s = join(
            testdata.models_dir, "maest", "discogs-maest-10s-pw-1.pb"
        )

        self.model30s = TensorflowPredictMAEST(graphFilename=self.graphFilename30s)
        self.model10s = TensorflowPredictMAEST(graphFilename=self.graphFilename10s)

    @classmethod
    def tearDownClass(self):
        essentia.log.warningActive = True

    def testRegression(self):
        expected = numpy.load(
            join(
                filedir(),
                "tensorflowpredictmaest",
                "preds_maest_discogs-maest-30s-pw-1.npy",
            )
        )

        filename = join(testdata.audio_dir, "recorded", "techno_loop.wav")
        audio = MonoLoader(filename=filename, sampleRate=16000)()

        activations = self.model30s(audio)
        found = numpy.mean(activations, axis=0).squeeze()

        self.assertAlmostEqualVector(found, expected, 1e-3)

    def testInvalidParam(self):
        self.assertConfigureFails(
            TensorflowPredictMAEST(),
            {
                "graphFilename": self.graphFilename30s,
                "batchSize": -2,
            },
        )  # Cannot be < -1.
        self.assertConfigureFails(
            TensorflowPredictMAEST(),
            {
                "graphFilename": self.graphFilename30s,
                "patchSize": 0,
            },
        )  # Cannot be 0.

    def testEmptyAudio(self):
        self.assertComputeFails(self.model30s, [])

    def testShortAudio(self):
        self.assertComputeFails(
            self.model30s, [1.0] * 16000 * 10
        )  # This model expects more than 30s of audio, 10s should fail.

    def testAutomaticPatchSizeConfig(self):
        found_patches = self.model10s([1.0] * 16000 * 20).shape[0]

        # 10s of audio, 5s of patch size -> 2 patches.
        self.assertEqual(found_patches, 2)


suite = allTests(TestTensorFlowPredictMAEST)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
