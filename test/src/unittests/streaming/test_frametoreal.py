#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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


class TestFrameToReal(TestCase):


    def testInvalidParam(self):
        self.assertConfigureFails(FrameToReal(), {'frameSize': -1})
        self.assertConfigureFails(FrameToReal(), {'hopSize': -1})
                # dimensions have to be different from 0.

    def testARealCase(self):
        filename = join(testdata.audio_dir, 'recorded', 'cat_purrrr.wav')

        ml = MonoLoader(filename=filename)
        fc = FrameCutter(frameSize=2048, hopSize=128)
        ftr = FrameToReal()
        fo = FileOutput()



        pool = Pool()
        ml.audio   >> fc.signal
        fc.frame   >> ftr.signal
        # FIXME How do deal with input and outputs having the same name? In this case signal.
        """
        ftr.signal   >> (pool, "signalIn")
        ftr.signal >> (pool, "signalOut")

        print(pool['signalIn'])
        print(pool['signalOut'])
        """

suite = allTests(TestFrameToReal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
