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

    def testRegression(self):
        # Test a simple tensorflow model trained on Essentia features.
        # The ground true values were obtained with the following script:
        expectedValues = [9.9997020e-01, 5.3647455e-14, 2.9801236e-05, 4.9495230e-12]

        patchSize = 128
        numberBands = 128
        frameSize = 1024
        hopSize = frameSize

        model = join(testdata.models_dir, 'vgg', 'vgg4.pb')
        filename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')

        audio = MonoLoader(filename=filename)()

        w = Windowing(type='hann', zeroPadding=frameSize)
        spectrum = Spectrum()
        mels = MelBands(numberBands=numberBands, type='magnitude')
        logNorm = UnaryOperator(type='log')

        bands = []
        for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
            melFrame = mels(spectrum(w(frame)))
            bands.append(logNorm(melFrame))
        bands = array(bands)

        discard = bands.shape[0] % patchSize
        bands = numpy.reshape(bands[:-discard,:], [-1, patchSize, numberBands])
        batch = numpy.expand_dims(bands, 1)

        pool = Pool()
        pool.set('model/Placeholder', batch)

        tfp = TensorflowPredict(graphFilename=model,
                                inputs=['model/Placeholder'],
                                outputs=['model/Softmax'],
                                isTraining=False,
                                isTrainingName='model/Placeholder_1')
        poolOut = tfp(pool)

        foundValues = poolOut['model/Softmax'].mean(axis=0).squeeze()

        self.assertAlmostEqualVector(foundValues, expectedValues, 1e-5)

    def testInvalidFilename(self):
        self.assertConfigureFails(TensorflowPredict(), {'graphFilename': ''})

    def testIdentityModel(self):
        # Perform the identity operation in Tensorflow to test if the data is
        # being copied correctly backwards and fordwards.
        model = join(filedir(), 'tensorflowpredict', 'identity.pb')
        filename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')

        audio = MonoLoader(filename=filename)()
        frames = array([frame for frame in FrameGenerator(audio)])
        batch = frames[numpy.newaxis, numpy.newaxis, :]

        pool = Pool()
        pool.set('model/Placeholder', batch)

        poolOut = TensorflowPredict(graphFilename=model,
                                    inputs=['model/Placeholder'],
                                    outputs=['model/Identity'])(pool)

        foundValues = poolOut['model/Identity']

        self.assertAlmostEqualMatrix(foundValues, batch)

suite = allTests(TestTensorFlowPredict)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
