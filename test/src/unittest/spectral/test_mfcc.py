#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

class TestMFCC(TestCase):

    def InitMFCC(self, numCoeffs):
        return MFCC(sampleRate = 44100,
                    numberBands = 40,
                    numberCoefficients = numCoeffs,
                    lowFrequencyBound = 0,
                    highFrequencyBound = 11000)


    def testRegression(self):
        # only testing that it yields valid result, but still need to check for
        # correct results.. no ground truth provided
        size = 20
        while (size > 0) :
            bands, mfcc = self.InitMFCC(size)(ones(1024))
            self.assertEqual(len(mfcc), size )
            self.assertEqual(len(bands), 40 )
            self.assert_(not any(numpy.isnan(mfcc)))
            self.assert_(not any(numpy.isinf(mfcc)))
            size -= 1

    def testZero(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1024
        val = amp2db(-90.0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 1 ):
            bands, mfcc = MFCC()(zeros(size))
            self.assertEqualVector(mfcc, expected)
            size /= 2

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
        from numpy import mean
        filename = join(testdata.audio_dir, 'recorded','musicbox.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frameGenerator = FrameGenerator(audio, frameSize=1024, hopSize=512)
        window = Windowing(type="blackmanharris62")
        pool=Pool()
        mfccAlgo = self.InitMFCC(13)

        for frame in frameGenerator:
            bands, mfcc = mfccAlgo(window(frame))
            pool.add("bands", bands)
            pool.add("mfcc", mfcc)

        expected = [-9.20311523e+02, 5.88643570e+01, -4.10509377e+01, 2.33458538e+01, 
                    -1.43523417e+01, 8.78132343e+00, -5.37768316e+00, 3.02709007e+00, 
                    -1.77758980e+00, 1.12805307e+00, -5.64552069e-01, 2.59827942e-01, 
                    5.14490485e-01]
    
        self.assertAlmostEqualVector(mean(pool['mfcc'], 0), expected, 1.0e-5)


suite = allTests(TestMFCC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
