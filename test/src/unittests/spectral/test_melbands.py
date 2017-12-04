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
import numpy as np

class TestMelBands(TestCase):

    def InitMelBands(self, nbands):
        return MelBands(inputSize=1024,
                        numberBands=nbands,
                        lowFrequencyBound=0,
                        highFrequencyBound=44100*.5)

    def testRegression(self):
        spectrum = [1]*1024
        mbands = self.InitMelBands(24)(spectrum)
        self.assertEqual(len(mbands), 24 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*24, 1e-5)

        mbands = self.InitMelBands(128)(spectrum)
        self.assertEqual(len(mbands), 128 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*128, 1e-5)

    def testRegressionHtkMode(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                                            sampleRate = 44100)()*2**15
        expected = [ 10.35452019,  12.97260263,  13.87114479,  12.92819811,  13.53927989,
                     13.65001411,  13.7067006,   12.72165126,  12.16052112,  12.29371287,
                     12.49577573,  12.68672873,  13.08112941,  12.59404232,  11.71325616,
                     11.48389955,  12.27751253,  12.07873884,  12.02260756,  12.42848721,
                     11.13694966,  10.49976274,  11.3370437,   12.06821492,  12.1631667,
                     11.84755549]

        frameSize = 1102
        hopSize = 441
        fftsize = 2048
        paddingSize = fftsize - frameSize
        spectrumSize = int(fftsize/2) + 1
        w = Windowing(type = 'hamming', 
                      size = frameSize, 
                      zeroPadding = paddingSize,
                      normalized = False,
                      zeroPhase = False)

        spectrum = Spectrum(size = fftsize)

        mbands = MelBands(inputSize= spectrumSize,
                          type = 'magnitude',
                          highFrequencyBound = 8000,
                          lowFrequencyBound = 0,
                          numberBands = 26,
                          warpingFormula = 'htkMel',
                          weighting = 'linear', 
                          normalize = 'unit_max')

        pool = Pool()
        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero = True, validFrameThresholdRatio = 1):
            pool.add('melBands', mbands(spectrum(w(frame))))

        self.assertAlmostEqualVector( np.mean(np.log(pool['melBands']),0), expected,1e-2)    


    def testZero(self):
        # Inputting zeros should return zero. Try with different sizes
        size = 1024
        while (size >= 256 ):
            self.assertEqualVector(MelBands()(zeros(size)), zeros(24))
            size /= 2

    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(MelBands(), [])
        self.assertComputeFails(MelBands(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(MelBands(), { 'numberBands': 0 })
        self.assertConfigureFails(MelBands(), { 'numberBands': 1 })
        self.assertConfigureFails(MelBands(), { 'lowFrequencyBound': -100 })
        self.assertConfigureFails(MelBands(), { 'lowFrequencyBound': 100,
                                                'highFrequencyBound': 50 })
        self.assertConfigureFails(MelBands(), { 'highFrequencyBound': 30000,
                                                'sampleRate': 22050})

    def testWrongInputSize(self):
        # This test makes sure that even though the inputSize given at
        # configure time does not match the input spectrum, the algorithm does
        # not crash and correctly resizes internal structures to avoid errors.
        spec = [.1,.4,.5,.2,.1,.01,.04]*100
        self.assertAlmostEqualVector(
                MelBands(inputSize=1024, sampleRate=10, highFrequencyBound=4)(spec),
                [ 0.0677984729,  0.0675897524,  0.0672208071,  0.0671572164,  0.0671814233,
                  0.0675696954,  0.0677656159,  0.0672371984,  0.0671650618,  0.0671633631,
                  0.0674924776,  0.0679418445,  0.067245841,   0.0671629086,  0.067130059,
                  0.0674537048,  0.0679484159,  0.0672798753,  0.0671581551,  0.0671161041,
                  0.0674532652,  0.0679644048,  0.067266494,   0.0671510249],
                1e-6)

    def testNotEnoughSpectrumBins(self):
        self.assertConfigureFails(MelBands(), {'numberBands': 256, 
                                               'inputSize': 1025})


suite = allTests(TestMelBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
