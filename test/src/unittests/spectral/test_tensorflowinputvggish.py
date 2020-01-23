#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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


class TestTensorflowInputVGGish(TestCase):

    def testRegression(self):
        # Hardcoded analysis parameters
        sampleRate = 16000
        frameSize = 400
        hopSize = 160

        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded/vignesh.wav'),
                           sampleRate=sampleRate)()

        expected = [ 9.18252861e-02,  1.11090327e+00,  1.27663188e+00,  7.67182742e-01,
                    -5.08587282e-01, -7.35953555e-01, -1.67525709e-01,  3.83205337e-01,
                     8.37841018e-01,  6.22906870e-01,  3.12534805e-03,  7.77315942e-01,
                     7.48971536e-01,  8.61180275e-01,  7.12877376e-02, -3.25823188e-01,
                    -3.45770341e-01, -6.06429663e-01, -1.23588742e+00, -8.00219610e-01,
                    -4.48695015e-01, -8.86346252e-01, -7.01429737e-01, -1.92338917e-01,
                    -5.24386627e-01, -4.86066723e-01,  2.13553896e-03, -4.40487141e-01,
                     1.96063574e-01,  4.46905819e-01, -8.60898118e-02, -5.08393019e-02,
                    -6.52852794e-01, -8.62813901e-01, -1.25070663e+00, -1.36270763e+00,
                    -1.52816302e+00, -1.39475057e+00, -1.02900690e+00, -4.27016926e-01,
                    -5.22283756e-01, -6.05011834e-01, -7.67452240e-01, -9.07988201e-01,
                    -1.04165164e+00, -8.13776139e-01, -4.83653929e-01, -2.00050489e-01,
                    -3.94115941e-01, -1.64469489e+00, -2.47999268e+00, -2.52071269e+00,
                    -2.31027679e+00, -2.20797295e+00, -2.03132993e+00, -1.53524443e+00,
                    -1.09347116e+00, -7.60057691e-01, -6.83884745e-01, -5.03377082e-01,
                    -4.72284172e-01, -7.03885687e-01, -1.11937559e+00, -1.25580341e+00]

        tfmf = TensorflowInputVGGish()
        frames = [tfmf(frame) for frame in FrameGenerator(audio,
                                                          frameSize=frameSize,
                                                          hopSize=hopSize,
                                                          startFromZero=True)]
        obtained = numpy.mean(frames, axis=0)

        # Setting a high tolerance value due to the mismatch between the
        # original and replicated features. However, they are close enough to
        # make valid predictions.
        self.assertAlmostEqualVector(obtained, expected, 1e1)

    def testInvalidInput(self):
        self.assertComputeFails(TensorflowInputVGGish(), [])

    def testWrongInputSize(self):
        # mel bands should fail for input size different to 400
        self.assertComputeFails(TensorflowInputVGGish(), [0.5] * 1)
        self.assertComputeFails(TensorflowInputVGGish(), [0.5] * 402)


suite = allTests(TestTensorflowInputVGGish)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
