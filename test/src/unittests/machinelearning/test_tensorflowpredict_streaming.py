#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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
from essentia.streaming import *
from essentia.standard import PoolAggregator


class TestTensorflowPredict_Streaming(TestCase):

    def identityModel(self, frameSize=1024, hopSize=512, patchSize=187,
                      lastPatchMode='discard'):
        # Identity test to check that the data flows properly.
        model = join(filedir(), 'tensorflowpredict', 'identity.pb')
        filename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')
        input_layer='model/Placeholder'
        output_layer='model/Identity'

        ml = MonoLoader(filename=filename)
        fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)
        vtt = VectorRealToTensor(shape=[1, 1, patchSize, frameSize],
                                 lastPatchMode=lastPatchMode)
        ttp = TensorToPool(namespace=input_layer)
        tfp = TensorflowPredict(graphFilename=model,
                                inputs=[input_layer],
                                outputs=[output_layer])
        ptt = PoolToTensor(namespace=output_layer)
        ttv = TensorToVectorReal()

        pool = Pool()

        ml.audio    >> fc.signal
        fc.frame    >> vtt.frame
        fc.frame    >> (pool, "framesIn")
        vtt.tensor  >> ttp.tensor
        ttp.pool    >> tfp.poolIn
        tfp.poolOut >> ptt.pool
        ptt.tensor  >> ttv.tensor
        ttv.frame   >> (pool, "framesOut")

        run(ml)

        return pool['framesOut'], pool['framesIn']

    def testRegression(self):
        # This test assesses the capability to work in streaming mode
        # and to be reset and process new files.
        model = join(testdata.models_dir, 'vgg', 'vgg4.pb')

        filenames = [join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav'),
                     join(testdata.audio_dir, 'recorded', 'distorted.wav'),
                     join(testdata.audio_dir, 'recorded', 'dubstep.wav'),
                     ]

        true_probs = [[1., 0., 0., 0.],
                      [0., 0., 0.005, 0.995],
                      [0., 0., 1., 0.]]

        frameSize = 1024
        hopSize = frameSize
        shape = [-1, 1, 128, 128]

        loader      = MonoLoader()
        w           = Windowing(type='hann', zeroPadding=frameSize)
        frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize,
                                  startFromZero=True)
        spectrum    = Spectrum()
        mel         = MelBands(numberBands=128, type='magnitude')
        logNorm     = UnaryOperator(type='log')
        vtt         = VectorRealToTensor(shape=shape,
                                              lastPatchMode='repeat',
                                              patchHopSize=1)
        ttp         = TensorToPool(mode='overwrite', namespace='model/Placeholder')
        tfp         = TensorflowPredict(graphFilename=model,
                                             inputs=['model/Placeholder'],
                                             outputs=['model/Softmax'],
                                             isTraining=False,
                                             isTrainingName='model/Placeholder_1')
        ptt         = PoolToTensor(namespace='model/Softmax')
        ttv         = TensorToVectorReal()

        pool = Pool()

        # Connecting the algorithms
        loader.audio      >> frameCutter.signal
        frameCutter.frame >> w.frame >> spectrum.frame
        spectrum.spectrum >> mel.spectrum
        mel.bands         >> logNorm.array >> vtt.frame
        vtt.tensor        >> ttp.tensor
        ttp.pool          >> tfp.poolIn
        tfp.poolOut       >> ptt.pool
        ptt.tensor        >> ttv.tensor
        ttv.frame         >> (pool, 'vgg_probs')

        aggrPool = PoolAggregator(defaultStats=['mean'])

        # Run Essentia
        for filename, expectedValues in zip(filenames, true_probs):
            loader.configure(filename=filename)
            run(loader)

            foundValues = aggrPool(pool)['vgg_probs.mean']

            reset(loader)
            pool.clear()

            self.assertAlmostEqualVector(foundValues, expectedValues, 1e-2)

    def testIdentityModel(self):
        # The test audio file has 430 frames.
        # Setting the patchSize to produce exactly 10 patches.
        numberOfFrames = 43
        found, expected = self.identityModel(patchSize=numberOfFrames,
                                             lastPatchMode='discard')
        self.assertAlmostEqualMatrix(found, expected, 1e-8)

        # Now the number of frames does not match an exact number of patches.
        # The expected output is trimmed to the found shape as with
        # lastPatchMode='discard' the remaining frames not fitting into a
        # patch are discarded.
        found, expected = self.identityModel(frameSize=256, hopSize=128,
                                             lastPatchMode='discard')
        self.assertAlmostEqualMatrix(found, expected[:found.shape[0], :], 1e-8)

        # Increase the patch size.
        found, expected = self.identityModel(frameSize=256, hopSize=128,
                                             patchSize=300, lastPatchMode='discard')
        self.assertAlmostEqualMatrix(found, expected[:found.shape[0], :], 1e-8)


suite = allTests(TestTensorflowPredict_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
