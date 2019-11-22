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

class TestTriangularBarkBands(TestCase):

    def InitTriangularBarkBands(self, nbands):
        return TriangularBarkBands(inputSize=1024,
                        numberBands=nbands,
                        lowFrequencyBound=0,
                        highFrequencyBound=44100*.5)

    def testRegression(self):
        spectrum = [1]*1024
        mbands = self.InitTriangularBarkBands(24)(spectrum)
        self.assertEqual(len(mbands), 24 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*24, 1e-5)

        mbands = self.InitTriangularBarkBands(128)(spectrum)
        self.assertEqual(len(mbands), 128 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*128, 1e-5)

    def testRegressionRastaMode(self):
        # Test the BFCC extractor compared to Rastamat specifications
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                                            sampleRate = 44100)()*2**15

        #Expected values generated in Rastamat/MATLAB
        expected = [ 20.28919141, 23.80362425, 26.69797305, 27.10461133, 26.64508125,
                     26.7758322, 27.1787682, 27.10699792, 26.29040982, 25.04243486, 
                     24.24791966, 24.17377063, 24.61976518, 25.29554584, 24.87617598, 
                     23.79018513, 23.04026225, 23.20707811, 23.09716777, 23.33050168,
                     22.8201923, 21.49477903, 21.63639095, 22.12937291, 22.01981441,
                     21.70728156]

        frameSize = 1102
        hopSize = 441
        fftsize = 2048
        paddingSize = fftsize - frameSize
        spectrumSize = int(fftsize/2) + 1
        w = Windowing(type = 'hann', 
                      size = frameSize, 
                      zeroPadding = paddingSize,
                      normalized = False,
                      zeroPhase = False)

        spectrum = Spectrum(size = fftsize)

        mbands = TriangularBarkBands(inputSize= spectrumSize,
                          type = 'power',
                          highFrequencyBound = 8000,
                          lowFrequencyBound = 0,
                          numberBands = 26,
                          weighting = 'linear', 
                          normalize = 'unit_max')

        pool = Pool()
        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero = True, validFrameThresholdRatio = 1):
            pool.add('TriangularBarkBands', mbands(spectrum(w(frame))))

        # Save results in a csv file. Use only for debugging purposes.
        # np.savetxt("out.csv", np.mean(np.log(pool['TriangularBarkBands']),0), delimiter=',')

        self.assertAlmostEqualVector( np.mean(np.log(pool['TriangularBarkBands']),0), expected,1e-2)


    def testZero(self):
        # Inputting zeros should return zero. Try with different sizes
        size = 1024
        while (size >= 256 ):
            self.assertEqualVector(TriangularBarkBands()(zeros(size)), zeros(24))
            size //= 2

    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(TriangularBarkBands(), [])
        self.assertComputeFails(TriangularBarkBands(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(TriangularBarkBands(), { 'numberBands': 0 })
        self.assertConfigureFails(TriangularBarkBands(), { 'numberBands': 1 })
        self.assertConfigureFails(TriangularBarkBands(), { 'lowFrequencyBound': -100 })
        self.assertConfigureFails(TriangularBarkBands(), { 'lowFrequencyBound': 100,
                                                'highFrequencyBound': 50 })
        self.assertConfigureFails(TriangularBarkBands(), { 'highFrequencyBound': 30000,
                                                'sampleRate': 22050})

    def testWrongInputSize(self):
        # This test makes sure that even though the inputSize given at
        # configure time does not match the input spectrum, the algorithm does
        # not crash and correctly resizes internal structures to avoid errors.
        spec = [.1,.4,.5,.2,.1,.01,.04]*100

        # Save results in a csv file. Use only for debugging purposes.
        # np.savetxt("out.csv", TriangularBarkBands(inputSize=1024, sampleRate=10, highFrequencyBound=4)(spec), delimiter=',')

        self.assertAlmostEqualVector(
                TriangularBarkBands(inputSize=1024, sampleRate=10, highFrequencyBound=4)(spec),
                [0.0460643246769905]*24,
                1e-6)

    """
    def testNotEnoughSpectrumBins(self):
        self.assertConfigureFails(TriangularBarkBands(), {'numberBands': 256, 
                                               'inputSize': 1025})
    """                                               


suite = allTests(TestTriangularBarkBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
