#!/usr/bin/env python

# Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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
from essentia import array as esarr
import numpy as np


class TestHumDetector(TestCase):

    # def testZero(self):
    #     self.assertEqual(HumDetector()(esarr(np.zeros(512)))[1], [])

    # def testOnes(self):
    #     self.assertEqual(HumDetector()(esarr(np.ones(512)))[1], [])

    def testInvalidParam(self):
        self.assertConfigureFails(HumDetector(), {'sampleRate': -1})
        self.assertConfigureFails(HumDetector(), {'frameSize': 0})
        self.assertConfigureFails(HumDetector(), {'hopSize': 0})
        self.assertConfigureFails(HumDetector(), {'Q0': -1})
        self.assertConfigureFails(HumDetector(), {'Q1': 2})
        self.assertConfigureFails(HumDetector(), {'minimumFrequency': 0})
        self.assertConfigureFails(HumDetector(), {'timeWindow': 0})

    def testSyntheticHum(self):
        import numpy as np

        filename = join(testdata.audio_dir, 'recorded/Vivaldi_Sonata_5_II_Allegro.wav')
        audio = MonoLoader(filename=filename)()

        nSamples = len(audio)

        time = np.linspace(0, nSamples / 44100., nSamples)

        freq = 50.

        hum = np.sin(2 * np.pi * freq * time )


        rHum, f, a, s = HumDetector(frameSize=0.4, hopSize=0.2)(np.array(audio + hum * 0.1, dtype=np.float32))

        self.assertAlmostEqualVector(f, [freq], 1e1)

suite = allTests(TestHumDetector)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
