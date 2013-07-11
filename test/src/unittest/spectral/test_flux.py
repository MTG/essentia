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


class TestFlux(TestCase):

    def testNotSameSize(self):
        # two consecutive calls with inputs of different sizes
        input1 = [3]*100
        input2 = [5]*101

        fluxAlgo = Flux()
        fluxAlgo(input1)
        self.assertComputeFails(fluxAlgo, input2)

    def testEmpty(self):
        self.assertAlmostEqual(Flux()([]), 0)

    def testSingle(self):
        self.assertAlmostEqual(Flux()([100]), 100)

    def testTwoIterations(self):
        input1 = [3]*100
        input2 = [3.1]*100

        fluxAlgo = Flux()
        self.assertAlmostEqual(fluxAlgo(input1), 30)
        self.assertAlmostEqual(fluxAlgo(input2), 1, 1e-6)

    def testZeros(self):
        input1 = [3.14]*50
        input2 = [0]*50
        input3 = [5.31]*50

        fluxAlgo = Flux()
        self.assertAlmostEqual(fluxAlgo(input1), 22.203152929, 1e-6)
        self.assertAlmostEqual(fluxAlgo(input2), 22.203152929, 1e-6)
        self.assertAlmostEqual(fluxAlgo(input3), 37.547370081, 1e-6)

    def testNegatives(self):
        # this is not a valid input as the magnitudes of a spectrum, but the
        # test will let us know when the behavior of the algorithm changes
        input1 = [-3]*100
        input2 = [-3.1]*100

        fluxAlgo = Flux()
        self.assertAlmostEqual(fluxAlgo(input1), 30)
        self.assertAlmostEqual(fluxAlgo(input2), 1, 1e-6)

    def testReset(self):
        input1 = [2.54]*100
        input2 = [6.21]*100

        fluxAlgo = Flux()
        self.assertAlmostEqual(fluxAlgo(input1), 25.4)
        fluxAlgo.reset()
        self.assertAlmostEqual(fluxAlgo(input2), 62.1, 1e-6)

    def testResetMismatchSize(self):
        input = [2.54]*100

        fluxAlgo = Flux()
        self.assertAlmostEqual(fluxAlgo(input), 25.4)
        fluxAlgo.reset()

        input = [2.54]*50
        self.assertAlmostEqual(fluxAlgo(input), 17.960512)

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded',
                        'mozart_c_major_30sec.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()

        fc = FrameCutter(frameSize=4096, hopSize=512)
        windower = Windowing(type='blackmanharris62')
        specAlg = Spectrum(size=4096)
        fluxAlg = Flux()

        # Calculate the average flux over all frames of audio
        frame = fc(audio)
        fluxSum = 0
        count = 0
        while len(frame) != 0:
            spectrum = specAlg(windower(frame))
            fluxSum += fluxAlg(spectrum)

            count += 1
            frame = fc(audio)

        fluxAvg = float(fluxSum) / float(count)
        self.assertAlmostEqual(fluxAvg, 0.0640958007244, 5e-5)


suite = allTests(TestFlux)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
