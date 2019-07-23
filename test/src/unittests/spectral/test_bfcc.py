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
from math import *
import numpy as np

class TestBFCC(TestCase):

    def InitBFCC(self, numCoeffs):
        return BFCC(inputSize = 1025,
                    sampleRate = 44100,
                    numberBands = 40,
                    numberCoefficients = numCoeffs,
                    lowFrequencyBound = 0,
                    highFrequencyBound = 11000)


    def testRegression(self):
        # only testing that it yields valid result, but still need to check for
        # correct results.. no ground truth provided
        size = 20
        while (size > 0) :
            bands, bfcc = self.InitBFCC(size)(ones(1025))
            self.assertEqual(len(bfcc), size )
            self.assertEqual(len(bands), 40 )
            self.assert_(not any(numpy.isnan(bfcc)))
            self.assert_(not any(numpy.isinf(bfcc)))
            size -= 1

    def testRegressionRastaMode(self):
        # Test the BFCC extractor compared to Rastamat specifications
        from numpy import mean
        frameSize = 1103
        hopSize = 441
        fftSize= 2048
        zeroPadding = fftSize - frameSize
        spectrumSize = int(fftSize/2) + 1

        #Expected values generated in Rastamat/MATLAB
        expected = array([ 174.2926614, 17.63766925, -12.01485808, -13.28523247,
                          -21.98320439, -27.65860448, -20.36286442, -11.31062965,
                          -3.406240974, -27.84575779, -27.37699052, -6.010315266, 
                          -14.08500904])


        #Scale into integer a la Rastamat/HTK
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                           sampleRate = 44100)()*2**15     

        w = Windowing(type = 'hann', 
                      size = frameSize, 
                      zeroPadding = zeroPadding,
                      normalized = False,
                      zeroPhase = False)

        spectrum = Spectrum(size = fftSize)

        bfccEssentia = BFCC(inputSize = spectrumSize,
                            type = 'power', 
                            weighting = 'linear',
                            lowFrequencyBound = 0,
                            highFrequencyBound = 8000,
                            numberBands = 26,
                            numberCoefficients = 13,
                            normalize = 'unit_max',
                            dctType = 3,
                            logType = 'log',
                            liftering = 22)

        pool = Pool()
        
        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize , startFromZero = True, validFrameThresholdRatio = 1):
            pool.add("bfcc",bfccEssentia(spectrum(w(frame)))[1])

        self.assertAlmostEqualVector( mean(pool['bfcc'], 0), expected ,1.0e-2)    


    def testZero(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = amp2db(0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256 ):
            bands, bfcc = BFCC()(zeros(size))
            self.assertEqualVector(bfcc, expected)
            size //= 2


    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(BFCC(), [])
        self.assertComputeFails(BFCC(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(BFCC(), { 'numberBands': 0 })
        self.assertConfigureFails(BFCC(), { 'numberBands': 1 })
        self.assertConfigureFails(BFCC(), { 'numberCoefficients': 0 })
        # number of coefficients cannot be larger than number of bands, otherwise dct
        # will throw and exception
        self.assertConfigureFails(BFCC(), { 'numberBands': 10, 'numberCoefficients':13 })
        self.assertConfigureFails(BFCC(), { 'lowFrequencyBound': -100 })
        # low freq bound cannot be larger than high freq bound
        self.assertConfigureFails(BFCC(), { 'lowFrequencyBound': 100,
                                            'highFrequencyBound': 50 })
        # high freq bound cannot be larger than half the sr
        self.assertConfigureFails(BFCC(), { 'highFrequencyBound': 30000,
                                            'sampleRate': 22050} )

    def testRealCase(self):
        # The expected values were recomputed from commit
        # 4f47eeb35f87fb6cb5ab9f184fbe03ab93cc4cd8
        from numpy import mean
        filename = join(testdata.audio_dir, 'recorded','musicbox.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frameGenerator = FrameGenerator(audio, frameSize=1025, hopSize=512)
        window = Windowing(type="blackmanharris62")
        pool=Pool()
        bfccAlgo = self.InitBFCC(13)

        for frame in frameGenerator:
            bands, bfcc = bfccAlgo(window(frame))
            pool.add("bands", bands)
            pool.add("bfcc", bfcc)

        expected = [-905.33917236,  53.90555191, -47.06598282, 30.50421333, -20.98745728,
                      14.16875744, -10.8211174,    7.4601965,  -5.96607971,   4.26305676,
                      -3.73092651,   2.7505765,   -2.49007034]

        self.assertAlmostEqualVector(mean(pool['bfcc'], 0), expected, 1.0e-6)


suite = allTests(TestBFCC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
