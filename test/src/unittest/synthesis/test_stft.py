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
    frames = cutFrames(params, signal)
    
    w = Windowing(type = "hann");
    fft = FFT(size = params['frameSize']);
    ifft = IFFT(size = params['frameSize']);
    overl = OverlapAdd (frameSize = params['frameSize'], hopSize = params['hopSize']);

    for f in frames:
      #outframe = OverlapAdd(frameSize = params['frameSize'], hopSize = params['hopSize'])(IFFT(size = params['frameSize'])(FFT(size = params['frameSize'])(Windowing()(f))))
      outframe = overl(ifft(fft(w(f))))
      outsignal = numpy.append(outsignal,outframe)
    
    return outsignal


class TestSTFT(TestCase):

    params = { 'frameSize': 2048, 'hopSize': 256, 'startFromZero': False }
#
#    def testZero(self):
#      
#        # generate test signal
#        signalSize = 10 * self.params['frameSize']
#        signal = zeros(signalSize)
#        
#        outsignal = analysisSynthesis(self.params, signal)
#        # cut to duration of input signal
#        outsignal = outsignal[:signalSize]
#
#        self.assertEqualVector(outsignal, signal)
#

#    def testWhiteNoise(self):
#        from random import random
#        # generate test signal
#        signalSize = 10 * self.params['frameSize']
#        signal = [2*(random()-0.5)*i for i in ones(signalSize)]
#        print max(signal), min(signal)
#
#        outsignal = analysisSynthesis(self.params, signal)
#        # cut to duration of input signal
#        outsignal = outsignal[:signalSize]
#        # compare without half-window bounds to avoid windowing effect
#        
#        numpy.savetxt('noise.txt',signal)
#        numpy.savetxt('noise_out.txt',outsignal)
#        
#        halfwin = (self.params['frameSize']/2)
#        self.assertAlmostEqualVector(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin])


#
    def testRegression(self):

        # generate test signal: sine 110Hz @44100kHz
        signalSize = 10 * self.params['frameSize']
        signal = 0.5 * numpy.sin( (array(range(signalSize))/44100.) * 110 * 2*math.pi)
    
      
        outsignal = analysisSynthesis(self.params, signal)
        # cut to duration of input signal
        outsignal = outsignal[:signalSize]
        # compare without half-window bounds to avoid windowing effect
        
        numpy.savetxt('sine.txt',signal)
        numpy.savetxt('sine_out.txt',outsignal)
        
        halfwin = (self.params['frameSize']/2)
        self.assertAlmostEqualVector(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin])





suite = allTests(TestSTFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

