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


def cleaningSineTracks(freqsTotal, minFrames):

  nFrames = freqsTotal.shape[0]
  begTrack = 0
  freqsClean = freqsTotal.copy()

  if (nFrames > 0 ):

    f = 0
    nTracks = freqsTotal.shape[1]# we assume all frames have a fix number of tracks

    for t in range (nTracks):

      f = 0
      begTrack = f

      while (f < nFrames-1):

        #// check if f is begin of track
        if (freqsClean[f][t] <= 0 and freqsClean[f+1][t] > 0 ):
          begTrack = f+1

        # clean track if shorter than min duration
        if ((freqsClean[f][t] > 0 and freqsClean[f+1][t] <= 0 ) and ( (f - begTrack) < minFrames)):
          for i in range(begTrack, f+1):
            freqsClean[i][t] = 0

        f+=1

  return freqsClean


def framesToAudio(frames):

    audio = frames.flatten()
    return audio


def analSprModelStreaming(params, signal):
    pool = essentia.Pool()
    fcut = es.FrameCutter(frameSize = params['frameSize'], hopSize = params['hopSize'], startFromZero =  False);
    w = es.Windowing(type = "blackmanharris92");
    spec = es.Spectrum(size = params['frameSize']);

    smanal = es.SprModelAnal(sampleRate = params['sampleRate'], maxnSines = params['maxnSines'], magnitudeThreshold = params['magnitudeThreshold'], freqDevOffset = params['freqDevOffset'], freqDevSlope = params['freqDevSlope'], minFrequency =  params['minFrequency'], maxFrequency =  params['maxFrequency'])

    # add half window of zeros to input signal to reach same ooutput length
    signal  = numpy.append(signal, essentia.zeros(params['frameSize'] // 2))
    insignal = VectorInput(signal)

    # analysis
    insignal.data >> fcut.signal

    fcut.frame >> smanal.frame
    smanal.magnitudes >> (pool, 'magnitudes')
    smanal.frequencies >> (pool, 'frequencies')
    smanal.phases >> (pool, 'phases')
    smanal.res >> (pool, 'res')

    essentia.run(insignal)

    # remove first half window frames
    mags = pool['magnitudes']
    freqs = pool['frequencies']
    phases = pool['phases']

    # remove short tracks
    minFrames = int( params['minSineDur'] * params['sampleRate'] / params['hopSize']);
    freqsClean = cleaningSineTracks(freqs, minFrames)

    return mags, freqsClean, phases


def analsynthSprModelStreaming(params, signal):
    pool = essentia.Pool()
    # windowing and FFT
    fcut = es.FrameCutter(frameSize = params['frameSize'], hopSize = params['hopSize'], startFromZero =  False)
    #w = es.Windowing(type = "blackmanharris92")
    #spec = es.Spectrum(size = params['frameSize'])

    smanal = es.SprModelAnal(sampleRate = params['sampleRate'], hopSize = params['hopSize'], maxnSines = params['maxnSines'], magnitudeThreshold = params['magnitudeThreshold'], freqDevOffset = params['freqDevOffset'], freqDevSlope = params['freqDevSlope'], minFrequency =  params['minFrequency'], maxFrequency =  params['maxFrequency'])
    synFFTSize = min(int(params['frameSize']/4), 4*params['hopSize'])  # make sure the FFT size is appropriate
    smsyn = es.SprModelSynth(sampleRate=params['sampleRate'],
                             fftSize=synFFTSize,
                             hopSize=params['hopSize'])

    # add half window of zeros to input signal to reach same ooutput length
    signal = numpy.append(signal, essentia.zeros(params['frameSize'] // 2))
    insignal = VectorInput(signal)

    # analysis
    insignal.data >> fcut.signal
    fcut.frame >> smanal.frame

    # synthesis
    smanal.magnitudes >> smsyn.magnitudes
    smanal.frequencies >> smsyn.frequencies
    smanal.phases >> smsyn.phases
    smanal.res >> smsyn.res

    smsyn.frame >> (pool, 'frames')
    smsyn.sineframe >> (pool, 'sineframes')
    smsyn.resframe >> (pool, 'resframes')

    essentia.run(insignal)

    outaudio = framesToAudio(pool['frames'])
    outaudio = outaudio[2*params['hopSize']:]

    return outaudio, pool


class TestSprModel(TestCase):

    params = { 'frameSize': 2048, 'hopSize': 128, 'startFromZero': False, 'sampleRate': 44100,'maxnSines': 100,'magnitudeThreshold': -74,'minSineDur': 0.02,'freqDevOffset': 10, 'freqDevSlope': 0.001, 'maxFrequency': 550.,'minFrequency': 65.}

    precisiondB = -40. # -40dB of allowed noise floor for sinusoidal model
    precisionDigits = int(-numpy.round(precisiondB/20.) -1) # -1 due to the rounding digit comparison.


    def testZero(self):

        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = essentia.zeros(signalSize)

        [mags, freqs, phases] = analSprModelStreaming(self.params, signal)

        # compare
        zerofreqs = numpy.zeros(freqs.shape)
        self.assertAlmostEqualMatrix(freqs, zerofreqs)


    def testWhiteNoise(self):
        from random import random
        # generate test signal
        signalSize = 10 * self.params['frameSize']
        signal = array([2*(random()-0.5)*i for i in essentia.ones(signalSize)])

        # for white noise test set sine minimum duration to 50ms, and min threshold of -20dB
        self.params['minSineDur'] = 0.35 # limit pitch tracks of a nimumim length of 350ms for the case of white noise input
        self.params['magnitudeThreshold']= -20

        [mags, freqs, phases]  = analSprModelStreaming(self.params, signal)

        # compare
        zerofreqs = numpy.zeros(freqs.shape)
        self.assertAlmostEqualMatrix(freqs, zerofreqs)


    def testRegression(self):

        # generate test signal: sine 110Hz @44100kHz
        signalSize = 10 * self.params['frameSize']
        signal = .5 * numpy.sin( (array(range(signalSize))/self.params['sampleRate']) * 110 * 2*math.pi)

        # generate noise components
        from random import random
        noise = 0.1 * array([2*(random()-0.5)*i for i in essentia.ones(signalSize)]) # -10dB
        signal = signal + noise

        outsignal, pool = analsynthSprModelStreaming(self.params, signal)

        outsignal = outsignal[:signalSize] # cut to durations of input and output signal

        # compare without half-window bounds to avoid windowing effect
        halfwin = (self.params['frameSize'] // 2)

        self.assertAlmostEqualVectorFixedPrecision(outsignal[halfwin:-halfwin], signal[halfwin:-halfwin], self.precisionDigits)


suite = allTests(TestSprModel)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
