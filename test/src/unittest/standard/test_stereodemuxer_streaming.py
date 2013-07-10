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
from essentia.streaming import AudioLoader, StereoDemuxer

class TestStereoDemuxer_Streaming(TestCase):

    def testRegression(self):
        size = 10
        input = [[size-i, i] for i in range(size)]
        mux = StereoDemuxer()
        gen = VectorInput(input)
        pool = Pool()

        gen.data >> mux.audio
        mux.left >> (pool, 'left')
        mux.right >> (pool, 'right')
        run(gen)

        self.assertEqualVector(pool['left'], [size-i for i in range(size)])
        self.assertEqualVector(pool['right'], [i for i in range(size)])

    def testEmpty(self):
        filename = join(testdata.audio_dir, 'generated', 'empty', 'empty.wav')
        loader = AudioLoader(filename=filename)
        mux = StereoDemuxer()
        #gen = VectorInput([{'left':None,'right':None} for i in range(10)])
        pool = Pool()

        loader.audio >> mux.audio
        loader.numberChannels >> (pool, 'channels')
        loader.sampleRate>> (pool, 'sampleRate')
        mux.left >> (pool, 'left')
        mux.right >> (pool, 'right')
        run(loader)

        self.assertRaises(KeyError, lambda: pool['left'])
        self.assertRaises(KeyError, lambda: pool['right'])


suite = allTests(TestStereoDemuxer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
