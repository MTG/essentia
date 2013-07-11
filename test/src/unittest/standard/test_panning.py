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
from math import log


def cutFrames(params, input):
    framegen = FrameGenerator(input,
                              frameSize = params['frameSize'],
                              hopSize = params['hopSize'],
                              startFromZero = params['startFromZero'])

    return [ frame for frame in framegen ]

class TestPanning(TestCase):

    def testRegression(self):
        # After comparing the results of panning with jordi janner's matlab
        # code, we have concluded that although they are not the same exact
        # numbersi, the algorithm seems to show correct output. Differences may
        # be due to essentia not being compiled for doubles, or may com from
        # slight differences in fft outputs. Window types and/or sizes or
        # normalization seem not to be critical for the final result.
        # On this respect, files computed with essentia at the time of this
        # writing (11/11/2008) have been included in order to have a regression
        # test that passes the test.

        # 10/07/2013: We switched to different testing audio files and recomputed 
        # the files with essentia assuming that Panning algo works correctly since 
        # it seems no fixes were ever done to it since the original test (11/11/2008)
        testdir = join(filedir(), 'panning')
        expected = readMatrix(join(testdir, 'essentia', 'musicbox_essentia_panning.txt'))

        framesize = 8192
        hopsize = 2048
        zeropadding = 1
        sampleRate = 44100

        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')

        left = MonoLoader(filename = filename, downmix = 'left', sampleRate = sampleRate)()
        right = MonoLoader(filename = filename, downmix = 'right', sampleRate = sampleRate)()

        frames_left = cutFrames({ 'frameSize': framesize, 'hopSize': hopsize, 'startFromZero': False },left)
        frames_right = cutFrames({ 'frameSize': framesize, 'hopSize': hopsize, 'startFromZero': False },right)

        spec = Spectrum()
        window = Windowing(size=framesize,
                           zeroPadding=framesize*zeropadding,
                           type = 'hann')
        panning = Panning(averageFrames=21) # matlab tests were generated with 21 (1second at 44100Hz)
        output = []
        
        for i in range(len(frames_left)):
            output = panning(spec(window(frames_left[i])), spec(window(frames_right[i])))
            # readVector messes up with the last digits, so for small numbers
            # we get errors above 1e-7: Is there a way to set precision in
            # python?
            self.assertAlmostEqualVectorFixedPrecision(expected[i], output[0], 2)

    def testMono(self):
        # checks that it works for mono signals loaded with audioloader, thus
        # right channel = 0
        inputSize = 512
        numCoeffs = 20
        specLeft = ones(inputSize)
        specRight = zeros(inputSize)
        panning = Panning(numCoeffs = numCoeffs)
        n = 0
        while n < 10:
            result = panning(specLeft, specRight)
            self.assertValidNumber(result.all())
            n += 1

    def testZero(self):
        inputSize = 512
        numCoeffs = 20
        expected = [-2.29359070e+02, -1.38243276e-03, -4.49713528e-01, 4.14732238e-03,
                     4.49690998e-01, -6.91174902e-03, -4.49645758e-01, 9.67658963e-03,
                     4.49589103e-01, -1.24400388e-02, -4.49509650e-01, 1.52042406e-02,
                     4.49419141e-01, -1.79666542e-02, -4.49305773e-01, 2.07295977e-02,
                     4.49181348e-01, -2.34902110e-02, -4.49033886e-01, 2.62518562e-02]
        spec = zeros(inputSize)
        panning = Panning(numCoeffs = numCoeffs)(spec, spec)
        self.assertAlmostEqualVector(panning[0], expected, 5e-4)


    def testEmpty(self):
        self.assertComputeFails(Panning(), [], [])

    def testInvalidParam(self):
        self.assertConfigureFails(Panning(), {'averageFrames': -1})
        self.assertConfigureFails(Panning(), {'panningBins': 0})
        self.assertConfigureFails(Panning(), {'numBands': 0})
        self.assertConfigureFails(Panning(), {'numCoeffs': 0})
        self.assertConfigureFails(Panning(), {'sampleRate': 0})

suite = allTests(TestPanning)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
