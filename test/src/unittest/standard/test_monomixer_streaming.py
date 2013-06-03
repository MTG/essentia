#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import MonoMixer, AudioLoader

class TestMonoMixer_Streaming(TestCase):
    left = []
    right = []

    def clickTrack(self):
        size = 100
        offset = 10
        self.left = [0]*size
        self.right = [0]*size
        for i in range(offset/2, size, offset):
            self.left[i] = 1.0
        for i in range(offset, size, offset):
            self.right[i] = 1

        output = []
        for i in range(size):
            output.append((self.left[i], self.right[i]))

        return array(output)

    def testLeft(self):
        gen = VectorInput(self.clickTrack())
        chGen = VectorInput([2])
        mixer = MonoMixer(type='left')
        pool = Pool()

        gen.data >> mixer.audio
        mixer.audio >> (pool, "mix")
        chGen.data >> mixer.numberChannels
        chGen.push('data', 2)
        run(gen)
        self.assertEqualVector(pool['mix'], self.left)

    def testRight(self):
        gen = VectorInput(self.clickTrack())
        chGen = VectorInput([2])
        mixer = MonoMixer(type='right')
        pool = Pool()

        gen.data >> mixer.audio
        mixer.audio >> (pool, "mix")
        chGen.data >> mixer.numberChannels
        chGen.push('data', 2)
        run(gen)
        self.assertEqualVector(pool['mix'], self.right)

    def testMix(self):
        gen = VectorInput(self.clickTrack())
        chGen = VectorInput([2])
        mixer = MonoMixer(type='mix')
        pool = Pool()

        gen.data >> mixer.audio
        mixer.audio >> (pool, "mix")
        chGen.data >> mixer.numberChannels
        chGen.push('data', 2)
        run(gen)
        self.assertEqual(sum(pool['mix']), 19*0.5)

    def testSingle(self):
        gen = VectorInput(array([(0.9, 0.5)]))
        chGen = VectorInput([2])
        mixer = MonoMixer(type='mix')
        pool = Pool()

        gen.data >> mixer.audio
        mixer.audio >> (pool, "mix")
        chGen.data >> mixer.numberChannels
        chGen.push('data', 2)
        run(gen)
        self.assertAlmostEqual(sum(pool['mix']), (0.9+0.5)*0.5)

    def testEmpty(self):
        inputFilename = join(testdata.audio_dir, 'generated', 'empty', 'empty.wav')
        loader = AudioLoader(filename=inputFilename)
        mixer = MonoMixer(type='left')
        pool = Pool()

        loader.audio >> mixer.audio
        mixer.audio >> (pool, "mix")
        loader.numberChannels >> mixer.numberChannels
        loader.sampleRate >> None
        run(loader)
        self.assertEqualVector(pool.descriptorNames(), [])

    def testInvalidParam(self):
        self.assertConfigureFails(MonoMixer(), {'type':'unknown'})


suite = allTests(TestMonoMixer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
