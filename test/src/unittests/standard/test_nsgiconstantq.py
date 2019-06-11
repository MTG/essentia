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
import essentia.standard as ess
import numpy as np

testdir = join(filedir(), 'nsgiconstantq')


class TestNSGIConstantQ(TestCase):

    def testSynthetiseSine(self):
        x = essentia.array(np.sin(2 * np.pi * 1000 * np.arange(2**12) / 44100))

        CQ, CQDC, DCNF = NSGConstantQ(inputSize=2**12)(x)

        # At the moment data needs to be transformed into a list of lists 
        CQList= list(CQ)
        for i in range(len(CQList)):
            CQList[i] = list(CQList[i]) 

        y = NSGIConstantQ(inputSize=2**12)(CQList, CQDC, DCNF)


        self.assertAlmostEqualVectorFixedPrecision(x, y, 3)

    def testSynthetiseSineLocalPhase(self):
        x = essentia.array(np.sin(2 * np.pi * 1000 * np.arange(2**12) / 44100))

        CQ, CQDC, DCNF= NSGConstantQ(inputSize=2**12,
                                                                phaseMode='local')(x)

        # At the moment data needs to be transformed into a list of lists 
        CQList= list(CQ)
        for i in range(len(CQList)):
            CQList[i] = list(CQList[i]) 

        y = NSGIConstantQ(inputSize=2**12, phaseMode='local')(CQList, CQDC, DCNF)


        self.assertAlmostEqualVectorFixedPrecision(x, y, 3)

    def testSynthetiseSineOddSize(self):
        # Test the reconstruction capabilities for signals with an odd length.
        inputSize = 2 ** 12 + 1
        x = essentia.array(np.sin(2 * np.pi * 1000 * np.arange(inputSize) / 44100))

        CQ, CQDC, DCNF= NSGConstantQ(inputSize=inputSize)(x)
        y = NSGIConstantQ(inputSize=inputSize)(CQ, CQDC, DCNF)

        self.assertAlmostEqualVectorFixedPrecision(x, y, 5)

    def testSynthetiseDC(self):
        x = essentia.array(np.ones(2**12))

        CQ, CQDC, DCNF = NSGConstantQ(inputSize=2**12)(x)
        CQList= list(CQ)
        for i in range(len(CQList)):
            CQList[i] = list(CQList[i]) 


        y = NSGIConstantQ(inputSize=2**12)(CQList, CQDC, DCNF)

        self.assertAlmostEqualVectorFixedPrecision(x, y, 1)

    def testInvalidParam(self):
        self.assertConfigureFails(NSGIConstantQ(), {'phaseMode': 'none'})
        self.assertConfigureFails(NSGIConstantQ(), {'inputSize': -1})
        self.assertConfigureFails(NSGIConstantQ(), {'inputSize': 0})
        self.assertConfigureFails(NSGIConstantQ(), {'minFrequency': 30000})
        self.assertConfigureFails(NSGIConstantQ(), {'minFrequency': 1000,
                                                            'maxFrequency': 500})
        self.assertConfigureFails(NSGIConstantQ(), {'maxFrequency': 0})
        self.assertConfigureFails(NSGIConstantQ(), {'binsPerOctave': 0})
        self.assertConfigureFails(NSGIConstantQ(), {'sampleRate': 0})  
        self.assertConfigureFails(NSGIConstantQ(), {'gamma': -1})  
        self.assertConfigureFails(NSGIConstantQ(), {'minimumWindow': 1})
        self.assertConfigureFails(NSGIConstantQ(), {'windowSizeFactor': 0})
        self.assertConfigureFails(NSGIConstantQ(), {'minimumWindow': 1})
    
    def testReconfigure(self):
        # The configuration of this algorithm is done the first time it is computed and 
        #  it will automatically change each time the input vectors modify their length. 

        x = essentia.array(np.sin(2 * np.pi * 1000 * np.arange(2**12) / 44100))
        CQ, CQDC, DCNF = NSGConstantQ(inputSize=2**12)(x)
        CQList= list(CQ)
        for i in range(len(CQList)):
            CQList[i] = list(CQList[i]) 

        nsgiconstantq = NSGIConstantQ()

        nsgiconstantq(CQList, CQDC, DCNF)

        # Reuse the algorith with different input shapes
        x = essentia.array(np.sin(2 * np.pi * 1000 * np.arange(2**13) / 44100))
        CQ, CQDC, DCNF = NSGConstantQ(inputSize=2**13)(x)
        CQList= list(CQ)
        for i in range(len(CQList)):
            CQList[i] = list(CQList[i]) 

        nsgiconstantq = NSGIConstantQ()
        
        nsgiconstantq(CQList, CQDC, DCNF)



suite = allTests(TestNSGIConstantQ)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
