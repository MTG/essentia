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


from numpy import *
import numpy as np # FIXME
from essentia_test import *

samplerate= 44100

class TestPitchMelodia(TestCase):
     

    def get_wave(self, freq, duration=0.2):
        '''
        Function takes the "frequecy" and "time_duration" for a wave 
        as the input and returns a "numpy array" of values at all points 
        in time
        '''
        amplitude = 1
        vector = vectorize(float)

        t = np.linspace(0, duration, int(samplerate * duration))
        wave=[]
        for i in range(int(samplerate * duration)):
            point = amplitude * sin(2 * np.pi * float(freq) * float(i))      
            wave.append(point)
        return(wave)

    def get_piano_notes(self):
        '''
        Returns a dict object for all the piano 
        note's frequencies
        '''
        # White keys are in Uppercase and black keys (sharps) are in lowercase
        octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
        base_freq = 261.63 #Frequency of Note C4
        
        note_freqs = {octave[i]: base_freq * pow(2,(i/12)) for i in range(len(octave))}        
        note_freqs[''] = 0.0 # silent note
        return note_freqs

    def get_song_data(self, music_notes):
        '''
        Function to concatenate all the waves (notes)
        '''
        note_freqs = self.get_piano_notes() # Function that we made earlier
        song = [self.get_wave(note_freqs[note]) for note in music_notes.split('-')]
        song = concatenate(song)
        return song

    # TODO This function not actually called yet in test code
    def testTwinkle(self):
        # Twinke twinkle little star.
        music_notes = 'C-C-G-G-A-A-G--F-F-E-E-D-D-C--G-G-F-F-E-E-D--G-G-F-F-E-E-D--C-C-G-G-A-A-G--F-F-E-E-D-D-C'
        data = self.get_song_data(music_notes)
        return(data)

    def testRootFifth(self):
        # first 5 notes
        music_notes = 'C-C-C-G-G-G'
        data = self.get_song_data(music_notes)
        return(data)

    def testZero(self):
        signal = zeros(1024)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testInvalidParam(self):
        self.assertConfigureFails(PitchMelodia(), {'binResolution': -1})
        self.assertConfigureFails(PitchMelodia(), {'filterIterations': 0})
        self.assertConfigureFails(PitchMelodia(), {'frameSize': -1})
        self.assertConfigureFails(PitchMelodia(), {'harmonicWeight': -1})
        self.assertConfigureFails(PitchMelodia(), {'hopSize': -1})        
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': -1})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeCompression': 2})
        self.assertConfigureFails(PitchMelodia(), {'magnitudeThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'minDuration': -1})
        self.assertConfigureFails(PitchMelodia(), {'minFrequency': -1})
        self.assertConfigureFails(PitchMelodia(), {'numberHarmonics': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakDistributionThreshold': 2.1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': -1})
        self.assertConfigureFails(PitchMelodia(), {'peakFrameThreshold': 2})                
        self.assertConfigureFails(PitchMelodia(), {'pitchContinuity': -1})                
        self.assertConfigureFails(PitchMelodia(), {'referenceFrequency': -1})             
        self.assertConfigureFails(PitchMelodia(), {'sampleRate': -1})
        self.assertConfigureFails(PitchMelodia(), {'timeContinuity': -1})

    def testOnes(self):
        signal = ones(1024)
        pitch, confidence = PitchMelodia()(signal)
        self.assertAlmostEqualVector(pitch, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertAlmostEqualVector(confidence, [0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def testEmpty(self):
        pitch, confidence = PitchMelodia()([])
        self.assertEqualVector(pitch, [])
        self.assertEqualVector(confidence, [])

    def testARealCase(self):
        frameSize = 1024
        sr = 44100
        hopSize = 512
        filename = join(testdata.audio_dir, 'recorded', 'vignesh.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()      
        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)
        #This code stores reference values in a file for later loading.
        save('pitchmelodiapitch.npy', pitch)             
        save('pitchmelodiaconfidence.npy', pitchConfidence)             

        loadedPitchMelodiaPitch = load(join(filedir(), 'pitchmelodia/pitchmelodiapitch.npy'))
        expectedPitchMelodiaPitch = loadedPitchMelodiaPitch.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitch, expectedPitchMelodiaPitch, 8)

        loadedPitchConfidence = load(join(filedir(), 'pitchmelodia/pitchmelodiaconfidence.npy'))
        expectedPitchConfidence = loadedPitchConfidence.tolist() 
        self.assertAlmostEqualVectorFixedPrecision(pitchConfidence, expectedPitchConfidence, 8)

    # TODO Check the outputs
    def testArtificalSong(self): 
        audio = self.testRootFifth()

        pm = PitchMelodia()
        pitch, pitchConfidence = pm(audio)
        #print(pitch, pitchConfidence)
        """
        FIXME

        The above print is outputing the first half of the "pitch" array as zeros
        and the second half with what appears to be the MIDI value 91 (G#6)
        This is not corresponding to the C-C-C-G-G-G   Root Fifth sequnce
        (also it is C4...G4 )

     
         84.333275 86.80444  88.32176  89.34802  89.86561
         90.909805 90.909805 91.43645  91.43645  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.43645  90.909805 89.34802  85.807396
          0.        0.        0.        0.        0.        0.        0.
         86.80444  89.86561  91.43645  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.43645  90.909805 89.34802  85.807396  0.
          0.        0.        0.        0.        0.        0.       87.3073
         90.3862   91.43645  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         91.96613  91.96613  91.96613  91.96613  91.96613  91.96613  91.96613
         92.498886 92.498886
    """

    def testArtificialSignal1(self):

        # generate test signal: sine 110Hz @44100kHz
        frameSize= 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin( (array(range(signalSize))/44100.) * 110 * 2*math.pi)
        pm = PitchMelodia()
        pitch, _ = pm(signal)
        index= int(len(pitch)/2) # Halfway point in pitch array
        self.assertAlmostEqual(pitch[50], 110.0,10)


    def testArtificialSignal2(self):

        # generate test signal: sine 1000Hz @44100kHz
        frameSize= 4096
        signalSize = 10 * frameSize
        signal = 0.5 * numpy.sin( (array(range(signalSize))/44100.) * 1000 * 2*math.pi)
        pm = PitchMelodia()
        pitch, _ = pm(signal)
        index= int(len(pitch)/2) # Halfway point in pitch array
        self.assertAlmostEqual(pitch[index], 1000.0,10)

suite = allTests(TestPitchMelodia)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
