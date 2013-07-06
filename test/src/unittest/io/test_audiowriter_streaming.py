#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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
from essentia.streaming import AudioWriter, AudioLoader
from essentia.streaming import MonoMixer
import os
from math import fabs

class TestAudioWriter_Streaming(TestCase):

    def compare(self, result, expected):
        self.assertEqual(len(result), len(expected))
        for i in range(len(result)):
            self.assertAlmostEqual(result[i][0] - expected[i][0], 0, 0.5/32767)
            self.assertAlmostEqual(result[i][1] - expected[i][1], 0, 0.5/32767)

    def testRegression(self):
        impulse = [-1, 0.3, 0.9999, 0.5, 0.2, 0.1]
        impulsePos = [0, 111, 5013, 20013, 11359, 44099]
        filename = "audiowritertest.wav"
        size = 44100
        signal = []
        for i in range(size):
            signal.append([0, 0])
        i = 0
        for pos in impulsePos:
            signal[pos][0] = impulse[i]
            signal[pos][1] = -impulse[i]
            i+=1

        # write the audio file:
        gen = VectorInput(signal)
        writer = AudioWriter(filename=filename)
        gen.data >> writer.audio
        run(gen)

        # load the audio file and validate:
        loader = AudioLoader(filename=filename)
        pool = Pool()

        loader.audio >> (pool, 'audio')
        loader.numberChannels >> None
        loader.sampleRate >> None
        run(loader)
        os.remove(filename)

        self.compare(pool['audio'], signal)

    def testEmpty(self):
        inputFilename = join(testdata.audio_dir, 'generated', 'empty', 'empty.wav')
        outputFilename = 'audiowritertest.wav'
        loader = AudioLoader(filename=inputFilename)
        pool = Pool()
        writer = AudioWriter(filename=outputFilename)
        loader.audio >> writer.audio
        loader.numberChannels >> None
        loader.sampleRate >> None
        run(loader)

        loader = AudioLoader(filename=outputFilename)
        loader.audio >> (pool, 'audio')
        loader.numberChannels >> None
        loader.sampleRate >> None
        run(loader)
        os.remove(outputFilename)

        self.assertEqualVector(pool.descriptorNames(), [])

    def testSaturation(self):
        impulse = [2, -2.3, 1.9999, -1.5, 1.2, 1.1]
        impulsePos = [0, 111, 5013, 20013, 11359, 44099]
        filename = "audiowritertest.wav"
        size = 44100
        signal = []
        expected = []
        for i in range(size):
            signal.append([0, 0])
            expected.append([0, 0])

        i = 0
        # strangely, negative values saturate at -1.0 - 1.0/32767.0
        # instead of -1.0 and have the positive saturate at 1.0-1/32767
        for pos in impulsePos:
            signal[pos][0] = impulse[i]
            signal[pos][1] = -impulse[i]
            if signal[pos][0] > 1:
                expected[pos][0] = 1
                expected[pos][1] = -1-1.0/32767.0
            else:
                expected[pos][0] =  -1-1.0/32767.0
                expected[pos][1] = 1
            i+=1

        # write the audio file:
        gen = VectorInput(signal)
        writer = AudioWriter(filename=filename)
        gen.data >> writer.audio
        run(gen)

        # load the audio file and validate:
        loader = AudioLoader(filename=filename)
        pool = Pool()

        loader.audio >> (pool, 'audio')
        loader.numberChannels >> None
        loader.sampleRate >> None
        run(loader)
        os.remove(filename)
        self.compare(pool['audio'], expected)

    def testOneSample(self):
        filename = "audiowritertest.wav"
        signal = array([[0.5, 0.3]])

        # write the audio file:
        gen = VectorInput(signal)
        writer = AudioWriter(filename=filename)
        gen.data >> writer.audio
        run(gen)

        # load the audio file and validate:
        loader = AudioLoader(filename=filename)
        pool = Pool()

        loader.audio >> (pool, 'audio')
        loader.numberChannels >> None
        loader.sampleRate >> None
        run(loader)
        os.remove(filename)
        self.compare(pool['audio'], signal)

####################################################
# the test below are not used cause we need to supply a specific ffmpeg with
# enabled mp3 and vorbis
####################################################

    def encoderTest(self, filename, bitrate=320, precision=1e-7):
        from math import sin, pi
        format = os.path.splitext(filename)[1].split('.')[1]
        sr = 44100
        sine = [i/44100.*sin(2.0*pi*10.0*i/sr) for i in xrange(sr)]
        #sine = [0.5*sin(2.0*pi*10.0*i/sr) for i in xrange(sr)]
        signal = array([[val,val] for val in sine])

        # write the audio file:
        gen = VectorInput(signal)
        writer = AudioWriter(filename=filename, format=format, bitrate=bitrate)
        gen.data >> writer.audio
        run(gen)

        # load the audio file and validate:
        loader = AudioLoader(filename=filename)
        mixer = MonoMixer(type='left')
        pool = Pool()
        loader.audio >> mixer.audio
        mixer.audio >> (pool, 'audio')
        loader.numberChannels >> mixer.numberChannels
        loader.sampleRate >> None
        run(loader)
        self.assertAlmostEqual(max(pool['audio']), max(sine), precision)
        from essentia.standard import ZeroCrossingRate
        zcr = int(ZeroCrossingRate(threshold=0.001)(pool['audio'])*len(pool['audio'])+0.5)
        expected = int(ZeroCrossingRate(threshold=0.0)(sine)*len(sine)+0.5)
        # for debugging:
        #from pylab import show, plot, figure
        from essentia.standard import MonoLoader
        #plot(sine)
        #plot(MonoLoader(filename=filename)())
        #show(figure)
        os.remove(filename)
        self.assertEqual(zcr, expected) # expected should be 20 (double the frequency)

    def atestMp3(self):
        self.encoderTest('audiowritertest.mp3', bitrate=320, precision=2e-4)

    def testWave(self):
        self.encoderTest('audiowritertest.wav', precision=5e-6)

    def testAiff(self):
        self.encoderTest('audiowritertest.aiff', precision=5e-6);

    def testflac(self):
        self.encoderTest('audiowritertest.flac', precision=5e-6);

    def testOgg(self):
        self.encoderTest('audiowritertest.ogg', precision=5e-6);


suite = allTests(TestAudioWriter_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
