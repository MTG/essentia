#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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




def analsynthHarmonicMaskStreaming(params, signal):
  
    out = array([0.])
  
    pool = essentia.Pool()
    # windowing and FFT
    fcut = es.FrameCutter(frameSize = params['frameSize'], hopSize = params['hopSize'], startFromZero =  False);
    w = es.Windowing(type = "blackmanharris92");
    fft = es.FFT(size = params['frameSize']);
    spec = es.Spectrum(size = params['frameSize']);
    
    # pitch detection
    pitchDetect = es.PitchYinFFT(frameSize=params['frameSize'], sampleRate =  params['sampleRate'])    
      
    hmask= es.HarmonicMask(sampleRate = params['sampleRate'], binWidth = params['binWidth'], attenuation = params['attenuation_dB'])
    
    ifft = es.IFFT(size = params['frameSize']);
    overl = es.OverlapAdd (frameSize = params['frameSize'], hopSize = params['hopSize']);

    
    # add half window of zeros to input signal to reach same ooutput length
    signal  = numpy.append(signal, zeros(params['frameSize'] // 2))
    insignal = VectorInput (signal)
        
      
    # analysis
    insignal.data >> fcut.signal
    fcut.frame >> w.frame
    w.frame >> spec.frame
    w.frame >> fft.frame
    spec.spectrum >> pitchDetect.spectrum
    
    fft.fft >> hmask.fft
    pitchDetect.pitch >> hmask.pitch  
    pitchDetect.pitchConfidence >> (pool, 'pitchConfidence')  

    hmask.fft >> ifft.fft
    
    ifft.frame >> overl.frame
    overl.signal >> (pool, 'audio')

    essentia.run(insignal)
    

    # remove first half window frames
    outaudio = pool['audio']
    outaudio = outaudio [2*params['hopSize']:]

    return outaudio, pool



#-------------------------------------

class TestHarmonicMask(TestCase):

    params = { 'frameSize': 2048, 'hopSize': 512, 'startFromZero': False, 'sampleRate': 44100,'binWidth': 2, 'attenuation_dB':-200}
    
    precisiondB = -40. # -40dB of allowed noise floor for sinusoidal model
    precisionDigits = int(-numpy.round(precisiondB/20.) -1) # -1 due to the rounding digit comparison.
    

    def testZero(self):
        
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = zeros(signalSize)
        
        outsignal,pool = analsynthHarmonicMaskStreaming(self.params, signal)

        outsignal = outsignal[:signalSize] # cut to durations of input and output signal

        # compare without half-window bounds to avoid windowing effect
        halfwin = (self.params['frameSize'] // 2)
              
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)



    def testRegression(self):

        # generate test signal: sine 20Hz @44100kHz
        signalSize = 10 * self.params['frameSize']
        signal = .5 * numpy.sin( (array(range(signalSize))/self.params['sampleRate']) * 220 * 2*math.pi)        
        
        outsignal,pool = analsynthHarmonicMaskStreaming(self.params, signal)

        outsignal = outsignal[:signalSize] # cut to durations of input and output signal

        # compare without half-window bounds to avoid windowing effect
        halfwin = (self.params['frameSize'] // 2)
              
        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)






suite = allTests(TestHarmonicMask)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

