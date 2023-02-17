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
import numpy as np

class TestNNLSChroma(TestCase):

    def testRegressionWithoutNNLS(self, frameSize=8192 + 1):
    # This test compares the Essentia implementation against the values obtained from the
    # NNLS Chroma VAMP plugin. In this case, the NNLS flag is deactivated. This option
    # should be also tested separately. However, right now activating this flag generates
    # a bigger numerical difference and  it makes no sense to create this test right now.
    # The source of this difference should be further investigated.
        expectedChroma = array([ 80.05112 ,  91.9131  , 119.02132 ,  78.63059 ,  90.60663 ,
                                 99.432625, 130.39139 , 121.17517 ,  93.71867 ,  69.87019 ,
                                 71.312325, 345.97983 ])
        expectedBassChroma = array([72.21934 , 68.03276 , 68.36602 , 46.983643, 66.85107 ,
                                    46.02778 , 53.00056 , 75.78875 , 87.5686  , 59.30848 ,
                                    71.43173 , 77.64267 ])

        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                    sampleRate = 44100)()

        w = Windowing(type='hann', normalized=False)
        spectrum = Spectrum()
        logspectrum = LogSpectrum(frameSize=frameSize)
        nnls = NNLSChroma(frameSize=frameSize, useNNLS=False)

        logfreqspectrogram = []
        for frame in FrameGenerator(audio, frameSize=16384, hopSize=2048,
                                    startFromZero=True):
            logfreqspectrum, meanTuning, _ = logspectrum(spectrum(w(frame)))
            logfreqspectrogram.append(logfreqspectrum)
        logfreqspectrogram = array(logfreqspectrogram)

        tunedLogfreqSpectrum, semitoneSpectrum, bassChroma, chroma =\
        nnls(logfreqspectrogram, meanTuning,  array([]))

        self.assertAlmostEqualVector(bassChroma.sum(axis=0), expectedBassChroma, 1e-1)
        self.assertAlmostEqualVector(chroma.sum(axis=0), expectedChroma, 1e-1)

    def testZero(self):
        # Inputting zeros should return zero. Try with different sizes
        size = 1024
        while (size >= 256 ):
            outs = NNLSChroma(frameSize = size)(array(zeros([2,256])),
                                                array(zeros(3)),
                                                array(zeros(2)))

            self.assertEqual(outs[0].sum() + outs[1].sum() + numpy.sum(outs[-2:]), .0)
            size = int(size/2)

    def testInvalidInput(self):
        self.assertComputeFails(NNLSChroma(), array([array([])]), zeros(3), zeros(2))
        self.assertComputeFails(NNLSChroma(), array([array([0.5])]), zeros(3), zeros(2))


    def testInvalidParam(self):
        self.assertConfigureFails(NNLSChroma(), { 'chromaNormalization': 'cosine' })
        self.assertConfigureFails(NNLSChroma(), { 'frameSize': 0 })
        self.assertConfigureFails(NNLSChroma(), { 'sampleRate': 0 })
        self.assertConfigureFails(NNLSChroma(), { 'spectralShape': 0 })
        self.assertConfigureFails(NNLSChroma(), { 'spectralWhitening': 2 })
        self.assertConfigureFails(NNLSChroma(), { 'tuningMode': 'none' })

    def testWrongInputSize(self):
        # This test makes sure that even though the frameSize given at
        # configure time does not match the input spectrum, the algorithm does
        # not crash and correctly resizes internal structures to avoid errors.
        print('\n')
        self.testRegressionWithoutNNLS(frameSize=1000)
        print('...')


suite = allTests(TestNNLSChroma)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
