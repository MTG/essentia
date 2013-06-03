#!/usr/bin/env python

from essentia_test import *
from essentia import *
from essentia.streaming import AudioOnsetsMarker, MonoWriter
from numpy import pi, sin
import os

file = "foo.wav"

class TestAudioOnsetsMarker_Streaming(TestCase):

    def testRegression(self):
        sr = 44100
        inputSize = sr # 1 second of audio
        input = [0.5*sin(2.0*pi*440.0*i/inputSize) for i in range(inputSize)]
        onsets = [0.15, 0.5, 0.9]

        signal = VectorInput(input)
        marker = AudioOnsetsMarker(sampleRate = sr, onsets=onsets)
        writer = MonoWriter(filename=file)
        signal.data >> marker.signal;
        marker.signal>> writer.audio;

        run(signal);

        left = MonoLoader(filename = file, downmix='left', sampleRate=sr)()
        diff = zeros(inputSize)
        for i in range(inputSize):
          # due to AudioOnsetsMarker downmixing by 0.5,
          # input[i] must be divided accordingly:
          diff[i] = left[i] - input[i]/2

        os.remove(file)
        onsetWidth = 0.04*sr
        epsilon = 1e-3
        found = []
        j = 0
        i = 0
        while i < inputSize:
          if diff[i] > epsilon:
            found.append(float(i)/float(sr))
            j += 1
            i += onsetWidth
          else: i+=1
        self.assertAlmostEqualVector(found, onsets, 1.5e-3)

    def testEmptySignal(self):
        sr = 44100
        signal = VectorInput([])
        onsets = [0.15, 0.5, 0.9]
        marker = AudioOnsetsMarker(sampleRate = sr, onsets=onsets)
        writer = MonoWriter(filename=file)
        signal.data >> marker.signal
        marker.signal >> writer.audio
        run(signal)
        self.assertTrue( not os.path.exists(file) )

    def testInvalidParam(self):
        self.assertConfigureFails(AudioOnsetsMarker(), { 'sampleRate' : 0 })
        self.assertConfigureFails(AudioOnsetsMarker(), { 'type' : 'burst' })
        self.assertConfigureFails(AudioOnsetsMarker(), { 'onsets' : [-1, -2, 9]})
        self.assertConfigureFails(AudioOnsetsMarker(), { 'onsets' : [1, -2, 9]})
        self.assertConfigureFails(AudioOnsetsMarker(), { 'onsets' : [2, 0, 9]})


suite = allTests(TestAudioOnsetsMarker_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
