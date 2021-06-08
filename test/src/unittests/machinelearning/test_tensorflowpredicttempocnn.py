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

class TestTensorflowPredictTempoCNN(TestCase):

    def testRegression(self):
        # The expected values were self-generated with TensorflowPredictTempoCNN
        # from commit deb334ba01f849e74b0e6e8554b2e17d73bc6966
        expected = [6.81436131e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 6.43281146e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.93597724e-06, 5.37369260e-06,
                    5.61165098e-06, 5.18454181e-06, 5.18454181e-06, 5.29615545e-06,
                    1.54262583e-04, 3.62297491e-04, 2.58193875e-04, 1.07488459e-05,
                    7.51755033e-06, 5.18454181e-06, 5.18454181e-06, 5.61612069e-06,
                    4.17900737e-05, 5.96406335e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.32490412e-06,
                    1.30493099e-05, 1.07832220e-05, 5.18454181e-06, 1.03671191e-05,
                    8.58808835e-06, 1.15191098e-04, 1.64165122e-05, 1.20488658e-05,
                    1.52187367e-05, 3.93088922e-05, 5.18454181e-06, 5.18454181e-06,
                    9.91457364e-06, 5.66988228e-06, 8.09343692e-06, 1.51625994e-04,
                    2.98513245e-04, 1.89710336e-05, 5.18454181e-06, 9.77961099e-06,
                    5.19856485e-06, 1.11890822e-05, 5.18454181e-06, 5.18454181e-06,
                    7.28983014e-06, 1.39451868e-05, 2.78414045e-05, 3.43431420e-05,
                    1.41270539e-05, 1.25369352e-05, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.27262637e-06, 7.14806265e-06,
                    6.32071078e-06, 5.98581255e-06, 2.18235818e-05, 1.09844186e-05,
                    1.34545080e-05, 5.18621164e-06, 1.68646511e-04, 1.09641778e-03,
                    1.38453615e-05, 3.01377531e-05, 1.21552097e-02, 9.64710116e-01,
                    1.37740122e-02, 1.28899017e-04, 5.89064148e-05, 8.77115599e-05,
                    1.34839304e-03, 4.11161163e-04, 2.86400136e-05, 5.18454181e-06,
                    1.44103651e-05, 2.40279369e-05, 5.18454181e-06, 5.48158596e-06,
                    3.18061357e-04, 1.57283881e-04, 1.29225233e-03, 2.19263566e-05,
                    5.95881011e-06, 5.19396053e-06, 5.18454181e-06, 6.97886162e-06,
                    5.99685018e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    1.38969726e-05, 1.19574506e-05, 5.18521047e-05, 6.30647337e-05,
                    6.18311378e-06, 5.18454181e-06, 5.56253462e-06, 1.13041697e-05,
                    2.63892180e-05, 1.03348422e-04, 1.33199101e-05, 1.43749730e-05,
                    3.64447420e-04, 2.36525921e-05, 9.88977990e-05, 8.33314698e-06,
                    5.58819511e-06, 1.15462044e-05, 5.68051291e-06, 3.33551179e-05,
                    1.42920344e-05, 1.12948746e-05, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.79629932e-06, 5.41440022e-06, 5.36711059e-06,
                    8.51361256e-06, 5.18454181e-06, 1.75346850e-05, 1.11004565e-05,
                    5.48184034e-05, 5.85014059e-05, 2.29484838e-04, 3.69668414e-05,
                    1.19517599e-05, 7.46400983e-05, 5.18454181e-06, 5.18454181e-06,
                    5.21284055e-05, 1.26784771e-05, 5.18454181e-06, 5.50738332e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.27911197e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 1.73420085e-05,
                    9.13620534e-06, 5.18454181e-06, 5.92750121e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.19091100e-06, 5.18454181e-06, 5.90774516e-06,
                    5.18454181e-06, 6.08426490e-06, 5.18454181e-06, 5.18454181e-06,
                    5.31059550e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.35843628e-06,
                    5.18454181e-06, 5.18454181e-06, 9.49293644e-06, 9.07244521e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 1.40652155e-05,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06,
                    5.18454181e-06, 5.18454181e-06, 5.18454181e-06, 5.18454181e-06]

        filename = join(testdata.audio_dir, 'recorded', 'techno_loop.wav')
        model = join(testdata.models_dir, 'tempocnn', 'deeptemp_k16.pb')

        audio = MonoLoader(filename=filename, sampleRate=11025)()
        found = TensorflowPredictTempoCNN(graphFilename=model)(audio)
        found = numpy.mean(found, axis=0)

        self.assertAlmostEqualVector(found, expected, 1e-6)

    def testEmptyInput(self):
        model = join(testdata.models_dir, 'tempocnn', 'deeptemp_k16.pb')

        self.assertRaises(RuntimeError, lambda: TensorflowPredictTempoCNN(graphFilename=model)([]))

    def testInvalidParam(self):
        model = join(testdata.models_dir, 'tempocnn', 'deeptemp_k16.pb')
        self.assertConfigureFails(TensorflowPredictTempoCNN(), {'graphFilename': model,
                                                                'batchSize': 0})
        self.assertConfigureFails(TensorflowPredictTempoCNN(), {'graphFilename': model,
                                                                'input': 'non_existing'})
        self.assertConfigureFails(TensorflowPredictTempoCNN(), {'graphFilename': model,
                                                                'output': 'non_existing'})
        self.assertConfigureFails(TensorflowPredictTempoCNN(), {'graphFilename': 'non_existing_model.pb',
                                                                'output': 'non_existing'})

suite = allTests(TestTensorflowPredictTempoCNN)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
