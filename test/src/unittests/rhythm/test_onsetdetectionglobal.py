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
from essentia_test import *
from essentia.standard import MonoLoader, OnsetDetectionGlobal as stdOnsetDetectionGlobal

framesize = 1024
hopsize = 512


class TestOnsetDetectionGlobal(TestCase):

    def testZero(self):
        # Inputting zeros should return no onsets(empty array)
        audio=zeros(44100*5)
        onset_beat_emphasis=OnsetDetectionGlobal(method='beat_emphasis')
        onset_infogain=OnsetDetectionGlobal(method='infogain')
        self.assertEqualVector(onset_beat_emphasis(audio), zeros(428))
        self.assertEqualVector(onset_infogain(audio), zeros(428))
     
    def testInvalidParam(self):
        self.assertConfigureFails(OnsetDetectionGlobal(), {'sampleRate':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(), {'method':'unknown'})
        self.assertConfigureFails(OnsetDetectionGlobal(), {'hopSize':-1})
        self.assertConfigureFails(OnsetDetectionGlobal(), {'frameSize':-1})

    def testRegression(self):
        audio = MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'techno_loop.wav'))()
        
        onsetdetectionglobal_infogain = stdOnsetDetectionGlobal(method='infogain')
        onsetdetectionglobal_beat_emphasis = stdOnsetDetectionGlobal(method = 'beat_emphasis')
        calculated_beat_emphasis = onsetdetectionglobal_infogain(audio).tolist()
        calculated_infogain = onsetdetectionglobal_beat_emphasis(audio).tolist()

        """
        This code stores reference values in a file for later loading.
        save('input_infogain.npy', calculated_beat_emphasis)
        save('input_beat_emphasis.npy', calculated_infogain)             
        """
        
        # Reference samples are loaded as expected values
        onsetdetectionglobal_infogain = load(join(filedir(), 'onsetdetectionglobal/infogain.npy'))
        onsetdetectionglobal_beat_emphasis = load(join(filedir(), 'onsetdetectionglobal/beat_emphasis.npy'))
        expected_infogain = onsetdetectionglobal_infogain.tolist()
        expected_beat_emphasis = onsetdetectionglobal_beat_emphasis.tolist()

        self.assertAlmostEqualVectorFixedPrecision(calculated_beat_emphasis, expected_beat_emphasis,2)
        self.assertAlmostEqualVectorFixedPrecision(calculated_infogain, expected_infogain,2)

    def testLongInputStreamingBufferResize(self):
        # Regression test for the output buffer overflow on long audio inputs
        # (RhythmExtractor2013/BeatTrackerMultiFeature failing with
        # "OnsetDetectionGlobal::onsetDetections: Could not push 1 value,
        # output buffer is full"). The streaming OnsetDetectionGlobal pushes
        # the whole detection function at once, and its output buffer used to
        # have a fixed capacity of 327680 values (~63 min of audio at the
        # default hopSize of 512). Use a small hopSize so that the number of
        # ODF frames exceeds the old capacity with a moderately-sized input
        # instead of an hour-long one (the buffer capacity is defined in
        # frames, independently of hopSize).
        from essentia.streaming import OnsetDetectionGlobal as strOnsetDetectionGlobal

        hopSize = 128
        frameSize = 256
        # yields 327690 ODF frames, just above the old capacity of 327680
        # (note: not 327681, as a number of frames equal to 1 modulo the
        # buffer's maxContiguousElements of 163840 triggers an unrelated
        # pre-existing PoolStorage issue mixing Pool::append and Pool::set)
        nSamples = 327691 * hopSize

        gen = VectorInput(zeros(nSamples))
        odg = strOnsetDetectionGlobal(method='infogain', frameSize=frameSize, hopSize=hopSize)
        p = Pool()

        gen.data >> odg.signal
        odg.onsetDetections >> (p, 'odf')

        # this used to raise RuntimeError: output buffer is full
        run(gen)

        self.assertTrue(len(p['odf']) > 327680)


suite=allTests(TestOnsetDetectionGlobal)

if __name__=='__main__':
    TextTestRunner(verbosity=2).run(suite)
    
