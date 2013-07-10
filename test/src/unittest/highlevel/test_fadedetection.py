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



from numpy import *
from essentia_test import *

framesize = 1024
hopsize = 512

class TestFadeDetection(TestCase):

    def testZero(self):
        # Inputting zeros should return no fades(empty array)
        sr = 44100
        size = int(10*sr/hopsize)
        rms = zeros(size)
        self.assertEqualMatrix(FadeDetection(frameRate=4)(zeros(size)), [[],[]])

    def testEmpty(self):
        # empty array should fail as mean cannot be computed
        self.assertComputeFails(FadeDetection(frameRate=4),([]))

    def testInvalidParam(self):
        self.assertConfigureFails(FadeDetection(), { 'frameRate': 0 })
        self.assertConfigureFails(FadeDetection(), { 'cutoffHigh': 0 })
        self.assertConfigureFails(FadeDetection(), { 'cutoffLow': 1 })


    def trapezium(self):
        sr = 44100
        frameRate = sr/hopsize # frames per second
        length = 20 # 20 seconds
        size = int(ceil(length*frameRate)) # size in (hop)frames
        fadelength = int(5.0*frameRate) # 5 second aprox
        cutoffHigh = 0.85
        cutoffLow = 0.2

        rms = zeros(size)
        for i in range(fadelength):
           rms[i] = 4.0*i/float(size)
        for i in range(fadelength, size-fadelength):
           rms[i] = 1.0
        for i in range(size-fadelength, size):
           rms[i] = 4.0*(size - i)/float(size)
        meanRms = mean(rms)
        fadeDetection = FadeDetection(frameRate=frameRate, cutoffHigh = cutoffHigh, cutoffLow = cutoffLow )
        foundFadeIn, foundFadeOut = fadeDetection(rms)

        fadeInStop = ceil(fadelength*cutoffHigh*meanRms)/frameRate
        fadeOutStart = length - fadeInStop
        expectedFadeIn = [[0, fadeInStop]]
        expectedFadeOut = [[fadeOutStart, length]]

        self.assertAlmostEqualMatrix(foundFadeIn, expectedFadeIn, 1e-3)
        self.assertAlmostEqualMatrix(foundFadeOut, expectedFadeOut, 1e-3)

    def doubleTrapezium(self):
        sr = 44100
        frameRate = sr/hopsize # frames per second
        length = 20 # 20 seconds
        size = int(ceil(length*frameRate)) # size in (hop)frames
        fadelength = int(5.0*frameRate) # 5 second aprox
        cutoffHigh = 0.85
        cutoffLow = 0.2

        # trapezoidal input:
        rms = [0]*size
        for i in range(fadelength):
           rms[i] = 4.0*i/float(size)
        for i in range(fadelength, size-fadelength):
           rms[i] = 1.0
        for i in range(size-fadelength, size):
           rms[i] = 4.0*(size - i)/float(size)
        rms += rms
        meanRms = mean(rms)
        fadeDetection = FadeDetection(frameRate=frameRate, cutoffHigh = cutoffHigh, cutoffLow = cutoffLow )
        foundFadeIn, foundFadeOut = fadeDetection(rms)

        fadeInStop = ceil(fadelength*cutoffHigh*meanRms)/frameRate
        fadeOutStart = length - fadeInStop
        expectedFadeIn = [[0, fadeInStop], [length, length+fadeInStop]]
        expectedFadeOut = [[fadeOutStart, length], [2*length - fadeInStop, 2*length]]

        self.assertAlmostEqualMatrix(foundFadeIn, expectedFadeIn, 1e-3)
        self.assertAlmostEqualMatrix(foundFadeOut, expectedFadeOut, 1e-3)

    def testRegression(self):
        self.trapezium()
        self.doubleTrapezium()


suite = allTests(TestFadeDetection)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
