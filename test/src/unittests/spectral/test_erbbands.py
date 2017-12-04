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

class TestERBBands(TestCase):

    def InitERBBands(self, nbands):
        return ERBBands(inputSize=1024,
                        numberBands=nbands,
                        lowFrequencyBound=0,
                        highFrequencyBound=44100*.5)

    def testRegression(self):
        spectrum = [1]*1024
        expected = [  1.45438981e+00,   1.81433212e+02,   2.04899731e+03,   8.41769434e+03,
                      2.25398730e+04,   4.75615469e+04,   8.64806875e+04,   1.42223219e+05,
                      2.17695609e+05,   3.15858281e+05,   4.39533906e+05,   5.91085625e+05,
                      7.71618000e+05,   9.79398250e+05,   1.20750662e+06,   1.43996650e+06,
                      1.64691088e+06,   1.78047738e+06,   1.77675975e+06,   1.57420688e+06,
                      1.15976588e+06,   6.31381562e+05,   1.99480609e+05,   1.94543945e+04]
        mbands = self.InitERBBands(24)(spectrum)
        self.assertEqual(len(mbands), 24 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

    def testZero(self):
        # Inputting zeros should return zero. Try with different sizes
        size = 1024
        while (size >= 256 ):
            self.assertEqualVector(ERBBands(inputSize = size)(zeros(size)), zeros(40))
            size = int(size/2)

    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(ERBBands(), [])
        self.assertComputeFails(ERBBands(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(ERBBands(), { 'numberBands': 0 })
        self.assertConfigureFails(ERBBands(), { 'numberBands': 1 })
        self.assertConfigureFails(ERBBands(), { 'lowFrequencyBound': -100 })
        self.assertConfigureFails(ERBBands(), { 'lowFrequencyBound': 100,
                                                'highFrequencyBound': 50 })
        self.assertConfigureFails(ERBBands(), { 'highFrequencyBound': 30000,
                                                'sampleRate': 22050})
        self.assertConfigureFails(ERBBands(), { 'width': 0 })
        
    def testWrongInputSize(self):
        # This test makes sure that even though the inputSize given at
        # configure time does not match the input spectrum, the algorithm does
        # not crash and correctly resizes internal structures to avoid errors.
        spec = [.1,.4,.5,.2,.1,.01,.04]*100
        self.assertAlmostEqualVector(
                ERBBands(inputSize=1024, sampleRate=10, highFrequencyBound=4, lowFrequencyBound=1, numberBands = 24)(spec),
                [ 47.16987228,  47.16987228,  47.16986847,  47.16987228,  47.16987228,
                  47.16987228,  47.16987228,  47.1698761,   47.1698761,   47.16987228,
                  47.16987228,  47.16987228,  47.16987228,  47.16987228,  47.16987228,
                  47.16987228,  47.16987228,  47.16987228,  47.16987228,  47.16986847,
                  47.16986847,  47.16986847,  47.16986847,  47.16986847],1e-6)



suite = allTests(TestERBBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
