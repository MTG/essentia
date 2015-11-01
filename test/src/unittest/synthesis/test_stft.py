#!/usr/bin/env python

# Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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
import math


def cutFrames(params, input = range(100)):

    if not 'validFrameThresholdRatio' in params:
      params['validFrameThresholdRatio'] = 0
      framegen = FrameGenerator(input,
                                  frameSize = params['frameSize'],
                                  hopSize = params['hopSize'],
                                  validFrameThresholdRatio = params['validFrameThresholdRatio'],
                                  startFromZero = params['startFromZero'])
                                  
    return [ frame for frame in framegen ]

def analysisSynthesis(params, signal):

    outsignal = array(0)
    # framecutter >  windowing > FFT > IFFT > OverlapAdd
    #    frames = cutFrames({ 'frameSize': framesize, 'hopSize': hopsize, 'startFromZero': False }, signal)
    frames = cutFrames(params, signal)
    for f in frames:
      outframe = OverlapAdd(frameSize = params['frameSize'], hopSize = params['hopSize'])(IFFT(size = params['frameSize'])(FFT(size = params['frameSize'])(Windowing()(f))))
      outsignal = numpy.append(outsignal,outframe)
    
    return outsignal


class TestSTFT(TestCase):


    def testZero(self):
      
        params = { 'frameSize': 2048, 'hopSize': 512, 'startFromZero': False }
        
        signalSize = 10 * params['frameSize'] # test duration
        signal = zeros(signalSize)
        
        outsignal = analysisSynthesis(params, signal)
        # cut to duration of input signal
        outsignal = outsignal[:signalSize]

        self.assertEqualVector(outsignal, signal)


#    def testWhiteNoise(self):
#        # input is [1, -1, 1, -1, ...] which corresponds to a sine of frequency Fs/2
#        sr = 44100.
#        dur = 0.2
#        signalSize = int(sr * dur)
#        fftSize = signalSize/2 + 1
#        
#        inputNyquist = ones(signalSize)
#          for i in range(signalSize):
#            if i % 2 == 1:
#              inputNyquist[i] = -1.0
#        
#          expected = [ 0+0j ] * fftSize
#          expected[-1] = (1+0j) * signalSize
#      
#        self.assertAlmostEqualVector(FFT()(inputNyquist), expected)
#  
#
#
#    def testRegression(self):
#      # use longer signal
#        signal = numpy.sin(numpy.arange(1024, dtype='f4')/1024. * 441 * 2*math.pi)
#        self.assertAlmostEqualVector(signal*len(signal), IFFT()(FFT()(signal)), 1e-2)
#
#        # readjust to our precision, otherwise 0.001 compared to 1e-12 would
#        # give a 1e9 difference...
#        for i, value in enumerate(expected):
#            if abs(value) < 1e-7:
#                expected[i] = 0+0j
#
#        self.assertAlmostEqualVector(FFT()(inputSignal), expected, 1e-2)




suite = allTests(TestSTFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

