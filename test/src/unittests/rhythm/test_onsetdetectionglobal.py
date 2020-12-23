#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the tebeat_emphasis of the GNU Affero General Public License as published by the Free
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

#infogain onset detection has been optimized for the default sampleRate=44100Hz, frameSize=2048, hopSize=512
#beat_emphasis is optimized for a fixed resolution of 11.6ms, which corresponds to the default sampleRate=44100Hz, frameSize=1024, hopSize=512
class TestOnsetDetectionGlobal(TestCase):

    def testZero(self):
        # Inputting zeros should return no onsets (empty array)
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/techno_loop.wav'),
                           sampleRate = 44100)()
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
        onset_infogain = OnsetDetectionGlobal(method='infogain')
        onset_beat_emphasis = OnsetDetectionGlobal(method='beat_emphasis')
        for frame in frames:
            zframe = zeros(len(frame))
            self.assertEqual(onset_infogain(zframe), 0)
            self.assertEqual(onset_beat_emphasis(zframe), 0)

    def testImpulse(self):
        # tests that for an impulse will yield the correct position
        audiosize = 10000
        audio = zeros(audiosize)
        pos = 5.5  # impulse will be in between frames 4 and 5
        audio[int(floor(pos*(hopsize)))] = 1.
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize,
                startFromZero=True)
        onset_beat_emphasis = OnsetDetectionGlobal(method='beat_emphasis')
        onset_infogain = OnsetDetectionGlobal(method='infogain')
        """
        nframe = 0
        for frame in frames:

            # 'beat_emphasis' (energy flux) and 'infogain' method will result in a non-zero value on frames 4 and 5,
            # energy flux for frame 6 is zero due to half-rectification
            # 'flux' on contrary will results in non-zero value for frame 6, as it does not half-rectify

            if nframe == floor(pos)-1:  # 4th frame
                self.assertNotEqual(onset_beat_emphasis(frame), 0)
                self.assertNotEqual(onset_infogain(frame), 0)
            elif nframe == ceil(pos)-1:  # 5th frame
                self.assertNotEqual(onset_complex_phase(frame, 0)
                self.assertNotEqual(onset_infogain(frame, 0)
            elif nframe == ceil(pos):  # 6th frame
                self.assertNotEqual(onset_flux(frame, 0)
                self.assertEqual(onset_infogain(frame, 0)
            else:
                self.assertEqual(onset_beat_emphasis(frame, 0)
                self.assertEqual(onset_infogain(frame, 0)
            nframe += 1
        """

    def testConstantInput(self):
        audio = ones(44100*5)
        frames = FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
        win = Windowing(type='hamming')
        fft = FFT()
        onset_infogain = OnsetDetectionGlobal(method='infogain')
        onset_beat_emphasis = OnsetDetectionGlobal(method='beat_emphasis')
        found_infogain = []
        found_beat_emphasis = []
        for frame in frames:
            found_infogain += [onset_infogain(frame)]
            found_beat_emphasis += [onset_beat_emphasis(frame)]
        print(zeros(len(found_infogain)))
        #self.assertEqualVector(found_infogain, zeros(len(found_infogain)))
        #self.assertEqualVector(found_beat_emphasis, zeros(len(found_beat_emphasis)))
    
    def testInvalidParam(self):
        self.assertConfigureFails(OnsetDetectionGlobal(), { 'sampleRate':-1 })
        self.assertConfigureFails(OnsetDetectionGlobal(), { 'method':'unknown' })

    def testComplexInputSizeMismatch(self):
        # Empty input should raise an exception
        signal = []
        self.assertComputeFails(OnsetDetectionGlobal(), signal)
        signal = ones(1024)
        self.assertComputeFails(OnsetDetectionGlobal(method='infogain'), signal)
        self.assertComputeFails(OnsetDetectionGlobal(method='beat_emphasis'), signal)
    


suite = allTests(TestOnsetDetectionGlobal)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
