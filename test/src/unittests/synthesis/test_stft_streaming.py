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
import essentia.streaming as es
import essentia.standard as std

def cutFrames(params, input = range(100)):

    if not 'validFrameThresholdRatio' in params:
      params['validFrameThresholdRatio'] = 0
    framegen = std.FrameGenerator(input,
                                frameSize = params['frameSize'],
                                hopSize = params['hopSize'],
                                validFrameThresholdRatio = params['validFrameThresholdRatio'],
                                startFromZero = params['startFromZero'])
                                
    return [ frame for frame in framegen ]

def analysisSynthesis(params, signal):

    outsignal = array(0)
    signal = numpy.append(signal, zeros(params['frameSize'] // 2))
    
    frames = cutFrames(params, signal)
    
    w = std.Windowing(type = "hann");
    fft = std.FFT(size = params['frameSize']);
    ifft = std.IFFT(size = params['frameSize']);    
    overl = std.OverlapAdd (frameSize = params['frameSize'], hopSize = params['hopSize']);
    counter = 0
    for f in frames:
      
      
      # STFT analysis
      infft = fft(w(f))
      # here we could apply spectral transformations
      outfft = infft
    
      # STFT synthesis
      ifftframe = ifft(outfft)
      of = ifftframe
      outframe = overl(of)
      
      if counter >= (params['frameSize']/(2*params['hopSize'])):
        outsignal = numpy.append(outsignal,outframe)

      counter += 1

    
    return outsignal

def analysisSynthesisStreaming(params, signal):

    out = numpy.array(0)
    pool = essentia.Pool()
    fcut = es.FrameCutter(frameSize = params['frameSize'], hopSize = params['hopSize'], startFromZero =  False);
    w = es.Windowing(type = "hann");
    fft = es.FFT(size = params['frameSize']);
    ifft = es.IFFT(size = params['frameSize']);
    overl = es.OverlapAdd (frameSize = params['frameSize'], hopSize = params['hopSize']);
    
    # add half window of zeros to input signal to reach same ooutput length
    signal  = numpy.append(signal, zeros(params['frameSize'] // 2))
    insignal = VectorInput (signal)
    insignal.data >> fcut.signal
    fcut.frame >> w.frame
    w.frame >> fft.frame
    fft.fft >> ifft.fft
    ifft.frame >> overl.frame
    overl.signal >> (pool, 'audio')
    
    
    essentia.run(insignal)
    
    # remove first half window frames
    outaudio = pool['audio']
    outaudio = outaudio [2*params['hopSize']:]
    return outaudio


class TestSTFT(TestCase):

    params = { 'frameSize': 1024, 'hopSize': 256, 'startFromZero': False }
    
    precisiondB = -80. # -60dB of allowed noise floor.
    precisionDigits = int(-numpy.round(precisiondB/20.) -1)

    def testZero(self):
      
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = zeros(signalSize)
        
        outsignal = analysisSynthesisStreaming(self.params, signal)
        # cut to duration of input signal
        outsignal = outsignal[:signalSize]

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)

    def testWhiteNoise(self):
        from random import random
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = array([2*(random()-0.5)*i for i in ones(signalSize)])

        outsignal = analysisSynthesisStreaming(self.params, signal)
        outsignal = outsignal[:signalSize] # cut to duration of input signal
        
        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)

    def testRamp(self):
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = 0.5 * array([float(2*i%signalSize)/signalSize for i in range(signalSize)])
          
        outsignal = analysisSynthesisStreaming(self.params, signal)
        outsignal = outsignal[:signalSize] # cut to duration of input signal          

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)

    def testRegression(self):

        # generate test signal: sine 110Hz @44100kHz
        signalSize = 10 * self.params['frameSize']
        signal = 0.5 * numpy.sin( (array(range(signalSize))/44100.) * 110 * 2*math.pi)
        
        # outsignal = analysisSynthesis(self.params, signal)
        outsignal = analysisSynthesisStreaming(self.params, signal)
        outsignal = outsignal[:signalSize] # cut to durations of input and output signal

        # compare without half-window bounds to avoid windowing effect
        halfwin = int(self.params['frameSize'] // 2)
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)





suite = allTests(TestSTFT)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

