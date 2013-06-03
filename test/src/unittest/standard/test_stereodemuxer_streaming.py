#!/usr/bin/env python

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
