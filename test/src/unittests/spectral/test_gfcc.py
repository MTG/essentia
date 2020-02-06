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

class TestGFCC(TestCase):

    def InitGFCC(self, numCoeffs, logType):
        return GFCC(inputSize = 1025,
                    sampleRate = 44100,
                    numberBands = 40,
                    numberCoefficients = numCoeffs,
                    silenceThreshold = 1e-9,
                    lowFrequencyBound = 0,
                    highFrequencyBound = 11000,
                    logType = logType)


    def testRegression(self):
        # only testing that it yields valid result, but still need to check for
        # correct results.. no ground truth provided
        size = 20
        logType = 'natural'
        while (size > 0) :
            bands, gfcc = self.InitGFCC(size, logType)(ones(1025))
            self.assertEqual(len(gfcc), size )
            self.assertEqual(len(bands), 40 )
            self.assert_(not any(numpy.isnan(gfcc)))
            self.assert_(not any(numpy.isinf(gfcc)))
            size -= 1


    def testZerodbAmp(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = amp2db(0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256 ):
            bands, gfcc = GFCC(inputSize=size)(zeros(size))
            self.assertEqualVector(gfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(gfcc).any() and not np.isinf(gfcc).any())    
            size = int(size/2)

    def testZerodbPow(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = pow2db(0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, gfcc = GFCC(inputSize=size, logType='dbpow')(zeros(size))
            self.assertEqualVector(gfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(gfcc).any() and not np.isinf(gfcc).any())    
            size = int(size/2)

    def testZeroLog(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = lin2log(0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, gfcc = GFCC(inputSize=size, logType='log')(zeros(size))
            self.assertEqualVector(gfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(gfcc).any() and not np.isinf(gfcc).any())
            size = int(size/2)

    def testZeroNatural(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = 0.
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, gfcc = GFCC(inputSize=size, logType='natural')(zeros(size))
            self.assertEqualVector(gfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(gfcc).any() and not np.isinf(gfcc).any())    
            size = int(size/2)

    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(GFCC(), [])
        self.assertComputeFails(GFCC(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(GFCC(), { 'numberBands': 0 })
        self.assertConfigureFails(GFCC(), { 'numberBands': 1 })
        self.assertConfigureFails(GFCC(), { 'numberCoefficients': 0 })
        # number of coefficients cannot be larger than number of bands, otherwise dct
        # will throw and exception
        self.assertConfigureFails(GFCC(), { 'numberBands': 10, 'numberCoefficients':13 })
        self.assertConfigureFails(GFCC(), { 'lowFrequencyBound': -100 })
        # low freq bound cannot be larger than high freq bound
        self.assertConfigureFails(GFCC(), { 'lowFrequencyBound': 100,
                                            'highFrequencyBound': 50 })
        # high freq bound cannot be larger than half the sr
        self.assertConfigureFails(GFCC(), { 'highFrequencyBound': 30000,
                                            'sampleRate': 22050} )
        self.assertConfigureFails(GFCC(), { 'logType': 'linear'} )



    def testRealCase(self):
        from numpy import mean
        filename = join(testdata.audio_dir, 'recorded','musicbox.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        fftSize = 2**11
        frameSize = 1025
        paddingSize = fftSize - frameSize
        frameGenerator = FrameGenerator(audio, frameSize=frameSize, hopSize=512)
        window = Windowing(size = frameSize,
                           zeroPadding = paddingSize,
                           type="blackmanharris62")
        spectrum = Spectrum()
        pool=Pool()
        gfccAlgo = self.InitGFCC(13,'dbamp')

        for frame in frameGenerator:
            bands, gfcc = gfccAlgo(spectrum(window(frame)))
            pool.add("bands", bands)
            pool.add("gfcc", gfcc)

        expected = [ -32.09788513,  108.49235535, -118.94884491,  -97.3769455 ,
        -84.4560318 ,  -42.14659119,  -53.62257767,  -33.75775909,
        -12.94090176,  -16.27241516,  -27.33692932,   -5.6728158 ,
         -3.05162358]

        self.assertAlmostEqualVector(mean(pool['gfcc'], 0), expected, 1.0e-5)


    def testLogType(self):
        frameSize = 1025
        spectrum = ones(frameSize) 

        logExpected = [ 74.33055115, -16.84531593,  -8.10999775,  -4.98932076,  -4.18623924,
                        -2.79512048,  -2.58836699,  -1.83708572,  -1.75375175,  -1.29282641,
                        -1.250772,    -0.94376731,  -0.92012668]

        dBExpected = [ 322.81347656,  -73.15826416,  -35.22127533,  -21.66834259,  -18.18061638,
                       -12.13905621,  -11.24114037,   -7.97836208,   -7.61645222,   -5.61467838,
                       -5.43204021,   -4.09873676,   -3.99605465]

        logBands = self.InitGFCC(13, 'log')(spectrum)[1]
        dbBands = self.InitGFCC(13, 'dbpow')(spectrum)[1]

        self.assertAlmostEqualVector(logBands, logExpected, 1.0e-5)
        self.assertAlmostEqualVector(dbBands, dBExpected, 1.0e-5)


suite = allTests(TestGFCC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
