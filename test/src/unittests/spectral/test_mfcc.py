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

class TestMFCC(TestCase):

    def InitMFCC(self, numCoeffs, log_type='dbamp', dct_type=2,
                 silenceThreshold=1e-10, liftering=0):
        return MFCC(inputSize = 1025,
                    sampleRate = 44100,
                    numberBands = 40,
                    numberCoefficients = numCoeffs,
                    lowFrequencyBound = 0,
                    highFrequencyBound = 11000,
                    dctType=dct_type,
                    logType=log_type,
                    silenceThreshold=silenceThreshold,
                    liftering=liftering)


    def testValidResults(self):
        # only testing that it yields valid result, but still need to check for
        # correct results.. no ground truth provided
        size = 20
        while (size > 0) :
            bands, mfcc = self.InitMFCC(size)(ones(1025))
            self.assertEqual(len(mfcc), size )
            self.assertEqual(len(bands), 40 )
            self.assert_(not any(numpy.isnan(mfcc)))
            self.assert_(not any(numpy.isinf(mfcc)))
            size -= 1

    def testRegression(self):
        # This test tries several combinations of the parameters dctType,
        # logType, liftering and silenceThreshold.
        # The rest of the parameters are directly inherited by MelBands and have
        # their own unit tests. The expected values were recomputed from commit
        # fb1b32b716eee2dc23af8219d402fe0ce80f5ef1
        spectrum = [0.1 * i for i in range(8)] * 128 + [0.5]

        # MFCC with default parameters.
        mfccs = self.InitMFCC(13, log_type='dbamp', dct_type=2)(spectrum)[1]
        expected = [-97.47665405, -2.33175659, -2.09009075, -1.78271627, -1.51546073,
                     -1.2559967,  -1.04854584, -0.83390617, -0.60241103, -0.44342852,
                     -0.48855162, -0.5175457,  -0.60503697 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with DCT type 3.
        mfccs = self.InitMFCC(13, log_type='dbamp', dct_type=3)(spectrum)[1]
        expected = [-137.85279846, -2.33175659, -2.09009075, -1.78271627, -1.51546073,
                      -1.2559967,  -1.04854584, -0.83390617, -0.60241103, -0.44342852,
                      -0.48855162, -0.5175457,  -0.60503697 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with log type 'natural'.
        mfccs = self.InitMFCC(13, log_type='natural', dct_type=2)(spectrum)[1]
        expected = [ 1.10026228, -0.00906091, -0.00920506, -0.00904771, -0.00949879, -0.0093013,
                    -0.00959581, -0.01000924, -0.01046225, -0.01158216, -0.01470083, -0.01487315,
                    -0.01388442 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with log type 'natural' and DCT type 3.
        mfccs = self.InitMFCC(13, log_type='natural', dct_type=3)(spectrum)[1]
        expected = [ 1.55600595, -0.00906091, -0.00920506, -0.00904771, -0.00949879, -0.0093013,
                    -0.00959581, -0.01000924, -0.01046225, -0.01158216, -0.01470083, -0.01487315,
                    -0.01388442 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with log type 'dbpow'.
        mfccs = self.InitMFCC(13, log_type='dbpow', dct_type=2)(spectrum)[1]
        expected = [-48.73832703, -1.1658783,  -1.04504538, -0.89135814, -0.75773036,
                     -0.62799835, -0.52427292, -0.41695309, -0.30120552, -0.22171426,
                     -0.24427581, -0.25877285, -0.30251849 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with log type 'dbpow' and DCT type 3.
        mfccs = self.InitMFCC(13, log_type='dbpow', dct_type=3)(spectrum)[1]
        expected = [-68.92639923, -1.1658783,  -1.04504538, -0.89135814, -0.75773036,
                     -0.62799835, -0.52427292, -0.41695309, -0.30120552, -0.22171426,
                     -0.24427581, -0.25877285, -0.30251849 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with log type 'log'.
        mfccs = self.InitMFCC(13, log_type='log', dct_type=2)(spectrum)[1]
        expected = [-11.22241306,  -0.26845309,  -0.24063081,  -0.20524287,  -0.17447406,
                    -0.1446017,   -0.12071839,  -0.09600687,  -0.06935519,  -0.05105186,
                    -0.05624655,  -0.05958462,  -0.0696575 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with log type 'log' and DCT type 3.
        mfccs = self.InitMFCC(13, log_type='log', dct_type=3)(spectrum)[1]
        expected = [-15.87088871, -0.26845309, -0.24063081, -0.20524287, -0.17447406,
                     -0.1446017,  -0.12071839, -0.09600687, -0.06935519, -0.05105186,
                     -0.05624655, -0.05958462, -0.0696575 ]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with 'silenceThreshold' = 1e-09.
        mfccs = self.InitMFCC(13, silenceThreshold=1e-9)(spectrum)[1]
        expected = [-97.47665405, -2.33175659, -2.09009075, -1.78271627, -1.51546073,
                     -1.2559967,  -1.04854584, -0.83390617, -0.60241103, -0.44342852,
                     -0.48855162, -0.5175457,  -0.60503697]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with 'silenceThreshold' = 1e-09 and 'liftering' = 22.
        mfccs = self.InitMFCC(13, silenceThreshold=1e-9, liftering=22)(spectrum)[1]
        expected = [-97.47665405, -5.98203564, -8.56740379, -9.92895412, -10.52797985,
                    -10.30352879, -9.76536465, -8.55069828, -6.63010693,  -5.12356043,
                     -5.80791903, -6.2105484,  -7.19270134]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)

        # MFCC with 'liftering' = 22.
        mfccs = self.InitMFCC(13, liftering=22)(spectrum)[1]
        expected = [-97.47665405, -5.98203564, -8.56740379, -9.92895412, -10.52797985,
                    -10.30352879, -9.76536465, -8.55069828, -6.63010693,  -5.12356043,
                     -5.80791903, -6.2105484,  -7.19270134]
        self.assertAlmostEqualVector(mfccs, expected, 1.e-6)


    def testRegressionHtkMode(self):
        from numpy import mean
        frameSize = 1102
        hopSize = 441
        fftSize= 2048
        zeroPadding = fftSize - frameSize
        spectrumSize = int(fftSize/2) + 1
        expected = array([88.78342438,   6.36632776,  -1.62886882,  -5.30124903,
                          -3.92886806, -13.52705765,  -7.62559938, -18.51092339,
                           2.24103594,   3.10329103, -16.25634193,  -5.46565485,
                         -14.56163502 ])

        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                           sampleRate = 44100)()*2**15

        w = Windowing(type = 'hamming',
                      size = frameSize,
                      zeroPadding = zeroPadding,
                      normalized = False,
                      zeroPhase = False)

        spectrum = Spectrum(size = fftSize)

        mfccEssentia = MFCC(inputSize = spectrumSize,
                            type = 'magnitude', 
                            warpingFormula = 'htkMel',
                            weighting = 'warping',
                            highFrequencyBound = 8000,
                            numberBands = 26,
                            numberCoefficients = 13,
                            normalize = 'unit_max',
                            dctType = 3,
                            logType = 'log',
                            liftering = 22)

        pool = Pool()

        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize , startFromZero = True, validFrameThresholdRatio = 1):
            pool.add("mfcc",mfccEssentia(spectrum(w(frame)))[1])

        self.assertAlmostEqualVector( mean(pool['mfcc'], 0), expected ,1e0)


    def testZerodbAmp(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = amp2db(0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, mfcc = MFCC(inputSize=size)(zeros(size))
            self.assertEqualVector(mfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(mfcc).any() and not np.isinf(mfcc).any())    
            size = int(size/2)

    def testZerodbPow(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = pow2db(0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, mfcc = MFCC(inputSize=size, logType='dbpow')(zeros(size))
            self.assertEqualVector(mfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(mfcc).any() and not np.isinf(mfcc).any())    
            size = int(size/2)

    def testZeroLog(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = lin2log(0)

        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, mfcc = MFCC(inputSize=size, logType='log')(zeros(size))
            self.assertEqualVector(mfcc, expected)

            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(mfcc).any() and not np.isinf(mfcc).any())    
            size = int(size/2)

    def testZeroNatural(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1025
        val = 0.
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 256):
            bands, mfcc = MFCC(inputSize=size, logType='natural')(zeros(size))
            self.assertEqualVector(mfcc, expected)
            
            # also assess that the thresholding is working
            self.assertTrue(not np.isnan(mfcc).any() and not np.isinf(mfcc).any())    
            size = int(size/2)

    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(MFCC(), [])
        self.assertComputeFails(MFCC(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(MFCC(), { 'numberBands': 0 })
        self.assertConfigureFails(MFCC(), { 'numberBands': 1 })
        self.assertConfigureFails(MFCC(), { 'numberCoefficients': 0 })
        # number of coefficients cannot be larger than number of bands, otherwise dct
        # will throw and exception
        self.assertConfigureFails(MFCC(), { 'numberBands': 10, 'numberCoefficients':13 })
        self.assertConfigureFails(MFCC(), { 'lowFrequencyBound': -100 })
        # low freq bound cannot be larger than high freq bound
        self.assertConfigureFails(MFCC(), { 'lowFrequencyBound': 100,
                                            'highFrequencyBound': 50 })
        # high freq bound cannot be larger than half the sr
        self.assertConfigureFails(MFCC(), { 'highFrequencyBound': 30000,
                                            'sampleRate': 22050} )

    def testRealCase(self):
        # The expected values were recomputed from commit
        # a97a01a11509802665d8211f679637dd60d85f3e
        from numpy import mean
        filename = join(testdata.audio_dir, 'recorded','musicbox.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frameGenerator = FrameGenerator(audio, frameSize=1025, hopSize=512)
        window = Windowing(type="blackmanharris62")
        pool=Pool()
        mfccAlgo = self.InitMFCC(13)

        for frame in frameGenerator:
            bands, mfcc = mfccAlgo(window(frame))
            pool.add("bands", bands)
            pool.add("mfcc", mfcc)

        expected = [-925.62939453,  65.30918121, -47.7603302,  29.60293007, -20.25293922,
                      14.30178356, -10.4140501,    7.66825676, -5.91072369,   4.87522268,
                      -3.87032604,   3.17340517,  -2.13934493 ]


        self.assertAlmostEqualVector(mean(pool['mfcc'], 0), expected, 1.e-8)


suite = allTests(TestMFCC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
