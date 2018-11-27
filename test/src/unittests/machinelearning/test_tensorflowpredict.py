#!/usr/bin/env python

# Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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
        # https://github.com/MTG/essentia-web-demo/blob/master/asc_models/deploy_vgg.py
        # Note that failing this test does not necessary mean there is a
        # problem with TensorflowPredict as it depends on many other Essentia algorithms.   
        true_probs = numpy.array([1., 0., 0., 0.])
        true_max = 0.99999785
        model = join(filedir(), 'tensorflowpredict', 'vgg4.pb')
        filename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')

        loader = essentia.standard.MonoLoader(filename=filename)
        audio = loader()

        w = Windowing(type='hann', zeroPadding=1024)
        spectrum = Spectrum(size=1024)
        mels = MelBands(numberBands=128, type='magnitude')
        logNorm = UnaryOperator(type='log')

        log_mel_spectrogram = []

        for frame in FrameGenerator(audio, frameSize=1024, hopSize=1024, startFromZero=True):
            mel_frame = mels(spectrum(w(frame)))
            log_mel_spectrogram.append(logNorm(mel_frame))

        audio_rep = essentia.array(log_mel_spectrogram)

        # limiting the input audios to 20 sec
        # (not to stress our cheap server)
        audio_rep = audio_rep[:1000]

        length = audio_rep.shape[0]
        n_frames = 128
        if length < n_frames:  # repeat-pad
            audio_rep = numpy.squeeze(audio_rep)
            src_repeat = audio_rep
            while src_repeat.shape[0] < n_frames:
                src_repeat = numpy.concatenate((src_repeat, audio_rep), axis=0)    
                audio_rep = src_repeat
                audio_rep = audio_rep[:n_frames, :]

        last_frame = int(audio_rep.shape[0]) - n_frames + 1
        batch = numpy.expand_dims(audio_rep[0:n_frames, :], axis=0)
        for time_stamp in range(1, last_frame, 1):
            patch = numpy.expand_dims(audio_rep[time_stamp:time_stamp+n_frames, :], axis=0)
            batch = numpy.concatenate((batch, patch), axis=0)

        pool = Pool()
        pool.set('model/Placeholder', batch)

        tfp = TensorflowPredict(graphFilename=model,
                                inputs=['model/Placeholder'],
                                outputs=['model/Softmax'],
                                isTraining=False,
                                isTrainingName='model/Placeholder_1')

        poolOut = tfp(pool)

        probs = poolOut['model/Softmax']
        averaged_probs = probs.mean(axis=0).squeeze()

        self.assertAlmostEqualVector(averaged_probs, true_probs, 1e-1)
        self.assertAlmostEqual(averaged_probs.max(), true_max)

    def testInvalidFilename(self):
        self.assertComputeFails(TensorflowPredict(graphFilename=''), (Pool()))

suite = allTests(TestTensorFlowPredict)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
