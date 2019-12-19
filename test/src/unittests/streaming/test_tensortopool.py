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
from essentia.streaming import *


class TestTensorToPool(TestCase):

    def identityOperation(self, frameSize=1024, hopSize=512, patchSize=187,
                          lastPatchMode='discard'):
        # Identity test to check that the data flows properly
        filename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')
        namespace='tensor'

        ml = MonoLoader(filename=filename)
        fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)
        vtt = VectorRealToTensor(shape=[1, 1, patchSize, frameSize],
                                 lastPatchMode=lastPatchMode)
        ttp = TensorToPool(namespace=namespace)
        ptt = PoolToTensor(namespace=namespace)
        ttv = TensorToVectorReal()

        pool = Pool()

        ml.audio   >> fc.signal
        fc.frame   >> vtt.frame
        fc.frame   >> (pool, "framesIn")
        vtt.tensor >> ttp.tensor
        ttp.pool   >> ptt.pool
        ptt.tensor >> ttv.tensor
        ttv.frame  >> (pool, "framesOut")

        run(ml)

        return pool['framesOut'], pool['framesIn']

    def testFramesToPoolAndBackToFrames(self):
        # Patch size equal to number of frames
        numberOfFrames = 43
        found, expected = self.identityOperation(patchSize=numberOfFrames,
                                                 lastPatchMode='repeat')
        self.assertAlmostEqualMatrix(found, expected, 1e-8)

        # Default parameters
        found, expected = self.identityOperation(frameSize=256, hopSize=128,
                                                 lastPatchMode='repeat')
        self.assertAlmostEqualMatrix(found, expected[:found.shape[0], :], 1e-8)

        # Increse aquire size
        found, expected = self.identityOperation(frameSize=256, hopSize=128,
                                                 patchSize=300, lastPatchMode='repeat')
        self.assertAlmostEqualMatrix(found, expected[:found.shape[0], :], 1e-8)

    def testInvalidParam(self):
        self.assertConfigureFails(TensorToPool(), {'mode': ''})

suite = allTests(TestTensorToPool)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
