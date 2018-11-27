#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
import essentia.streaming as estr


class TestTensorflowPredict_Streaming(TestCase):

    def testRegression(self):
        # This test assesses the capability to work in streaming mode
        # and to be reset and process new files.
        model = join(filedir(), 'tensorflowpredict', 'vgg4.pb')

        filenames = [join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav'),
                     join(testdata.audio_dir, 'recorded', 'distorted.wav'),
                     join(testdata.audio_dir, 'recorded', 'dubstep.wav'),
                     ]

        true_probs = [[1., 0., 0., 0.],
                      [0., 0., 0.01, 0.99],
                      [0., 0., 1., 0.]
                      ]

        inputShape = [-1, 1, 128, 128]

        loader      = estr.MonoLoader(filename=filenames[0])
        w           = estr.Windowing(type='hann', zeroPadding=1024)
        frameCutter = estr.FrameCutter(frameSize=1024, hopSize=1024, startFromZero=True)
        spectrum    = estr.Spectrum(size=1024)
        mel         = estr.MelBands(numberBands=128, type='magnitude')
        logNorm     = estr.UnaryOperator(type='log')
        vtt         = estr.VectorRealToTensor(shape=inputShape,
                                              lastPatchMode='repeat',
                                              patchHopSize=1)
        ttp         = estr.TensorToPool(mode='overwrite', namespace='model/Placeholder')
        tfp         = estr.TensorflowPredict(graphFilename=model,
                                             inputs=['model/Placeholder'],
                                             outputs=['model/Softmax'],
                                             isTraining=False,
                                             isTrainingName='model/Placeholder_1')
        ptt         = estr.PoolToTensor(namespace='model/Softmax')
        ttv         = estr.TensorToVectorReal()
        fileout     = estr.FileOutput(filename='predictions.txt')

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
        for f, p in zip(filenames, true_probs):
            loader.configure(filename=f)

            essentia.run(loader)

            averaged_probs = aggrPool(pool)['vgg_probs.mean']

            self.assertAlmostEqualVector(averaged_probs, p, 1e-0)

            essentia.reset(loader)
            pool.clear()


suite = allTests(TestTensorflowPredict_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
