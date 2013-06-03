#!/usr/bin/env python

from essentia_test import *
from essentia import *

class TestStartStopSilence(TestCase):

    def runTest(self, input, expected):
        sss = StartStopSilence()
        for i in range(len(input)):
          result = sss(input[i])

        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def testNonSilent(self):
        nFrames = 10
        frameSize = 1024
        input = ones([nFrames,frameSize])
        self.runTest(input,[0, nFrames-1])

    def testSilent(self):
        nFrames = 10
        frameSize = 1024
        input = zeros([nFrames,frameSize])
        self.runTest(input,[nFrames-1, nFrames-1])

    def testSilent_NonSilent(self):
        nFrames = 10
        frameSize = 1024
        input = zeros([nFrames, frameSize])
        for i in range(3, nFrames):
            input[i] = ones(frameSize)
        self.runTest(input,[3, nFrames-1])

    def testNonSilent_Silent(self):
        nFrames = 10
        frameSize = 1024
        input = ones([nFrames, frameSize])
        for i in range(3, nFrames):
            input[i] = zeros(frameSize)
        self.runTest(input,[0, 2])

    def testSilent_NonSilent_Silent(self):
        nFrames = 10
        frameSize = 1024
        input = zeros([nFrames, frameSize])
        for i in range(3, nFrames-3):
            input[i] = ones(frameSize)
        self.runTest(input,[3, nFrames-3-1])

    def testSilent_NonSilent_Silent_NonSilent(self):
        nFrames = 10
        frameSize = 1024
        input = zeros([nFrames, frameSize])
        for i in range(3, 6):
            input[i] = ones(frameSize)
        for i in range(8, 10):
            input[i] = ones(frameSize)
        self.runTest(input,[3, nFrames-1])

    def testSilent_NonSilent_Silent_NonSilent_Silent(self):
        nFrames = 15
        frameSize = 1024
        input = zeros([nFrames, frameSize])
        for i in range(3, 6):
            input[i] = ones(frameSize)
        for i in range(8, 10):
            input[i] = ones(frameSize)
        self.runTest(input,[3, 9])

   # def testEmpty(self):
   #     sss = StartStopSilence()
   #     gen = VectorInput(zeros([10,0]))
   #     pool = Pool()

   #     gen.data >> sss.frame
   #     sss.startFrame >> (pool,'start')
   #     sss.stopFrame >> (pool, 'stop')
   #     self.assertRaises(EssentiaException, lambda: run(gen))

    def testOne(self):
        self.runTest(ones([1,1]), [0, 0])
        self.runTest(zeros([1,1]), [0, 0])

suite = allTests(TestStartStopSilence)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
