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

import essentia
#import essentia.streaming as es
import essentia.standard as std

counterrunner = 0

def cutFrames(params, input = range(100)):

    if not 'validFrameThresholdRatio' in params:
      params['validFrameThresholdRatio'] = 0
    framegen = std.FrameGenerator(input,
                                frameSize = params['frameSize'],
                                hopSize = params['hopSize'],
                                validFrameThresholdRatio = params['validFrameThresholdRatio'],
                                startFromZero = params['startFromZero'])
                                
    return [ frame for frame in framegen ]

def analysisSynthesisStandard(params, signal):
  
    w = std.Windowing(type = "hann");
    fft = std.FFT(size = params['frameSize']);
    ifft = std.IFFT(size = params['frameSize']);    
    overl = std.OverlapAdd (frameSize = params['frameSize'], hopSize = params['hopSize']);
    # add half window of zeros to input signal to reach same ooutput length
    signal  = numpy.append(signal, zeros(params['frameSize'] // 2))
    
    frames = cutFrames(params, signal)

    outsignal = []
    counter = 0
    outframe = array(0)
    for f in frames:
      
      outframe = overl(ifft(fft(w(f))))
      outsignal = numpy.append(outsignal,outframe)


    outsignal = outsignal [2*params['hopSize']:]
    return outsignal


class TestSTFT(TestCase):

    params = { 'frameSize': 1024, 'hopSize': 256, 'startFromZero': False }

    precisiondB = -80. # -60dB of allowed noise floor.
    precisionDigits = int(-numpy.round(precisiondB/20.) -1)

    global counterrunner
    counterrunner += 1

    def testZero(self):
      
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = zeros(signalSize)
        
        outsignal = analysisSynthesisStandard(self.params, signal)
        # cut to duration of input signal
        outsignal = outsignal[:signalSize]

#        numpy.savetxt('zeros.txt',signal)
#        numpy.savetxt('zeros_out.txt',outsignal)

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)


    def testWhiteNoise(self):
        from random import random
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = array([2*(random()-0.5)*i for i in ones(signalSize)])

        outsignal = analysisSynthesisStandard(self.params, signal)
        outsignal = outsignal[:signalSize] # cut to duration of input signal

#        numpy.savetxt('noise.txt',signal)
#        numpy.savetxt('noise_out.txt',outsignal)

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)

    def testRamp(self):
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = 0.5 * array([float(2*i%signalSize)/signalSize for i in range(signalSize)])

        outsignal = analysisSynthesisStandard(self.params, signal)
        outsignal = outsignal[:signalSize]  # cut to duration of input signal

#        numpy.savetxt('ramp.txt',signal)
#        numpy.savetxt('ramp_out.txt',outsignal)

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)

    def testRegression(self):

        # generate test signal: sine 110Hz @44100kHz
        signalSize = 10 * self.params['frameSize']
        signal = 0.5 * numpy.sin( (array(range(signalSize))/44100.) * 110 * 2*math.pi)
        
        outsignal = analysisSynthesisStandard(self.params, signal)
        outsignal = outsignal[:signalSize] # cut to durations of input and output signal

#        numpy.savetxt('sine.txt',signal)
#        numpy.savetxt('sine_out.txt',outsignal)

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)





suite = allTests(TestSTFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

