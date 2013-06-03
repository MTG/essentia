#!/usr/bin/env python

from essentia_test import *
from essentia.standard import AudioLoader, StereoDemuxer

class TestStereoDemuxer_Streaming(TestCase):

    def testRegression(self):
        size = 10
        input = array([[size-i, i] for i in range(size)])
        left, right = StereoDemuxer()(input)

        self.assertEqualVector(left, [size-i for i in range(size)])
        self.assertEqualVector(right , [i for i in range(size)])

    def testEmpty(self):
        filename = join(testdata.audio_dir, 'generated', 'empty', 'empty.wav')
        audio, _, _= AudioLoader(filename=filename)()
        left, right = StereoDemuxer()(audio)
        self.assertEqualVector(left , [])
        self.assertEqualVector(right , [])


suite = allTests(TestStereoDemuxer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
