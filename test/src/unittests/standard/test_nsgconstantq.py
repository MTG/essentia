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

testdir = join(filedir(), 'nsgconstantq')


class TestNSGConstantQ(TestCase):

    def initNsgconstantq(self, inputSize=2048, fmin=27, fmax=10000, binsPerOctave=24, normalize='none'):
        return NSGConstantQ(inputSize=inputSize,
                    minFrequency=fmin,
                    maxFrequency=fmax,
                    binsPerOctave=binsPerOctave,
                    sampleRate=44100,
                    rasterize='full',
                    phaseMode='global',
                    gamma=0,
                    normalize=normalize,
                    window='hannnsgcq',
                    )

    def testRegression(self):
        input = essentia.array(np.sin(2 * np.pi * 1000 * np.arange(2048) / 44100))
        # Compared against the implementation of the MATLAB CQT_toolbox_2013 
        expected = np.array([ 0.01764389 +8.19244758e-06j, -0.00327444 +1.78957267e-03j,
                             -0.00379942 +1.00535053e-02j,  0.00479218 +8.65996905e-03j,
                              0.00636455 -1.14715385e-03j, -0.00165716 -6.73704576e-03j,
                             -0.00948407 +1.39929814e-03j, -0.00132517 +9.10799044e-03j,
                              0.00804364 +2.70849478e-03j,  0.00373902 -8.13302867e-03j,
                             -0.00733613 -6.00933843e-03j, -0.00738841 +5.56821084e-03j,
                              0.00371405 +8.43253605e-03j,  0.00904939 -1.72925594e-03j,
                              0.00034281 -9.21268760e-03j, -0.00891524 -2.47832619e-03j,
                             -0.00459810 +8.25670810e-03j,  0.00651840 +6.09559784e-03j,
                              0.00661061 -5.63534139e-03j, -0.00441447 -8.19178966e-03j,
                             -0.00905809 +1.89702405e-03j,  0.00139695 +6.62663074e-03j,
                              0.00708779 -1.61311132e-03j,  0.00229181 -9.95998412e-03j,
                             -0.00574295 -7.79506339e-03j, -0.00166257 +5.33548630e-04j])
        
        output = np.mean(self.initNsgconstantq(normalize='sine')(input)[0],axis=0)

        self.assertAlmostEqualVector(np.abs(expected), np.abs(output), 1e-6)

    def testDC(self):
        # Checks the DC component of the transform
        input= essentia.array(np.ones(2**11))
        # Second output of NSGConstantQ contains the DC information nedeed for the inverse transform.
        DCfilter = self.initNsgconstantq()(input)[1]
        # Integrates the energy. DC filter should contain all the energy of the signal in this case.
        DCenergy = np.sum(DCfilter)
        inputEnergy = np.sum(input)
        self.assertEqual(inputEnergy , DCenergy)

    def testNyquist(self):
        inputSize = 2**11
        signalNyquist = [-1,  1] * int(inputSize / 2)

        CQ, DC, Nyquist = self.initNsgconstantq(inputSize=inputSize)(signalNyquist)

        # Checks that all the energy is contained in the Nyquist band
        self.assertEqual(np.sum(np.abs(CQ)), 0)
        self.assertEqual(np.sum(np.abs(DC)), 0)
        self.assertGreater(np.sum(np.abs(Nyquist)), 0)

    def testZero(self):
        inputSize = 2**11
        signalZero = [0] * inputSize
        output = np.abs(np.mean(self.initNsgconstantq()(signalZero)[0]))
        self.assertEqual(0, output)

    def testEmpty(self):
        # Checks whether an empty input vector yields an exception
        self.assertComputeFails(self.initNsgconstantq(),  [])

    def testOne(self,normalize='none'):
        # Checks for a single value
        self.assertComputeFails(self.initNsgconstantq(),  [1])

    def testInvalidParam(self):
        self.assertConfigureFails(self.initNsgconstantq(), {'inputSize': -1})
        self.assertConfigureFails(self.initNsgconstantq(), {'inputSize': 0})
        self.assertConfigureFails(self.initNsgconstantq(), {'minFrequency': 30000})
        self.assertConfigureFails(self.initNsgconstantq(), {'minFrequency': 1000,
                                                            'maxFrequency': 500})
        self.assertConfigureFails(self.initNsgconstantq(), {'maxFrequency': 0})
        self.assertConfigureFails(self.initNsgconstantq(), {'binsPerOctave': 0})
        self.assertConfigureFails(self.initNsgconstantq(), {'sampleRate': 0})  
        self.assertConfigureFails(self.initNsgconstantq(), {'gamma': -1})  
        self.assertConfigureFails(self.initNsgconstantq(), {'minimumWindow': 1})
        self.assertConfigureFails(self.initNsgconstantq(), {'windowSizeFactor': 0})
        self.assertConfigureFails(self.initNsgconstantq(), {'minimumWindow': 1})

    def testOddInput(self):
        # Checks that compute does not fail for even input (former behavior).
        a = np.ones(4099, dtype='float32')
        NSGConstantQ()(a)

    
suite = allTests(TestNSGConstantQ)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
