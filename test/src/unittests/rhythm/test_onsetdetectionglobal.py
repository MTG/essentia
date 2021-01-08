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

from numpy import *
from essentia_test import *

framesize = 1024
hopsize = 512

class TestOnsetDetectionGlobal(TestCase):
    
    def testZero(self):
       # Inputting zeros should return no onsets(empty array)
       audio=MonoLoader(filename=join(testdata.audio_dir,'recorded/techno_loop.wav'),
            sampleRate=44100)()
       frames=FrameGenerator(audio,frameSize=framesize,hopSize=hopsize)
       win=Windowing(type='hamming')
       fft=FFT()
       onset_beat_emphasis=OnsetDetectionGlobal(method='beat_emphasis')
       onset_infogain=OnsetDetectionGlobal(method='infogain')
       for frame in frames:
          fft_frame=fft(win(frame))
          mag,ph=CartesianToPolar()(fft_frame)
          mag=zeros(len(mag))
          self.assertEqual(onset_beat_emphasis(mag),0)
          self.assertEqual(onset_infogain(mag),0)
                
    def testImpulse(self):
       # tests that for an impulse will yield the correct position
        audiosize=10000
        audio=zeros(audiosize)
        pos=5.5#impulse will be in between frames 4 and 5
        audio[int(floor(pos*(hopsize)))]=1.
        frames=FrameGenerator(audio,frameSize=framesize,hopSize=hopsize,startFromZero=True)
        win=Windowing(type='hamming',zeroPadding=framesize)
        fft=FFT()
        onset_infogain=OnsetDetectionGlobal(method='infogain')
        onset_beat_emphasis=OnsetDetectionGlobal(method='beat_emphasis')
        nframe=0
        for frame in frames:
            mag,ph=CartesianToPolar()(fft(win(frame)))
            if nframe==floor(pos)-1:#4thframe
                self.assertEqual(onset_infogain(mag),0)
                self.assertAlmostEqual(onset_beat_emphasis(mag),0)
            elif nframe==ceil(pos)-1:#5thframe
                self.assertNotEqual(onset_infogain(mag),0)
                self.assertNotEqual(onset_beat_emphasis(mag),0)
            elif nframe==ceil(pos):#6thframe
                self.assertEqual(onset_infogain(mag),0)
                self.assertNotEqual(onset_beat_emphasis(mag),0)
            else:
                print(onset_infogain(mag))
                self.assertEqual(numpy.ndarray.tolist(onset_infogain(mag)),[0.])
                self.assertEqual(onset_beat_emphasis(mag),0.)
                nframe+=1


    def testConstantInput(self):
        audio=ones(44100*5)
        frames=FrameGenerator(audio,frameSize=framesize,hopSize=hopsize)
        win=Windowing(type='hamming')
        fft=FFT()
        onset_beat_emphasis=OnsetDetectionGlobal(method='beat_emphasis')
        onset_infogain=OnsetDetectionGlobal(method='infogain')
        found_beat_emphasis=[]
        found_infogain=[]
        for frame in frames:
            fft_frame=fft(win(frame))
            mag,ph=CartesianToPolar()(fft_frame)
            mag=zeros(len(mag))
            found_beat_emphasis+=[onset_beat_emphasis(mag)]
            found_infogain+=[onset_infogain(mag)]
            self.assertEqualVector(found_beat_emphasis,zeros(len(found_beat_emphasis)))
            self.assertEqualVector(found_infogain,zeros(len(found_infogain)))
    
    def testInvalidParam(self):
        self.assertConfigureFails(OnsetDetectionGlobal(),{'sampleRate':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(),{'method':'unknown'})
    

suite=allTests(TestOnsetDetectionGlobal)

if __name__=='__main__':
    TextTestRunner(verbosity=2).run(suite)
