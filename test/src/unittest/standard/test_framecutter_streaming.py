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
import essentia
import essentia.streaming as es
import essentia.standard as std

class TestFrameCutter_Streaming(TestCase):

    def cutFrames(self, options, input=range(100)):
        input = [float(x) for x in input]
        gen = VectorInput(input)
        pool = Pool()
        if not 'validFrameThresholdRatio' in options:
            options['validFrameThresholdRatio'] = 0

        frameCutter = es.FrameCutter(frameSize = options['frameSize'],
                                     hopSize = options['hopSize'],
                                     startFromZero = options['startFromZero'],
                                     validFrameThresholdRatio = options['validFrameThresholdRatio'])

        gen.data >> frameCutter.signal
        frameCutter.frame >> (pool, 'frame')
        run(gen)
        if pool.descriptorNames(): return pool['frame']
        return []

    def testInvalidoptions(self):
        self.assertConfigureFails(es.FrameCutter(), {'frameSize' : 0})
        self.assertConfigureFails(es.FrameCutter(), {'frameSize' : -23})
        self.assertConfigureFails(es.FrameCutter(), {'hopSize' : 0})
        self.assertConfigureFails(es.FrameCutter(), {'hopSize' : -23})

    def testEmpty(self):
        options = {'frameSize'     :100,
                   'hopSize'       : 60,
                   'startFromZero' : False}
        result = self.cutFrames(options, [])
        self.assertEqualVector(result, [])

    def testEmptyCentered(self):
        options = {'frameSize'     :100,
                   'hopSize'       : 60,
                   'startFromZero' : True}
        result = self.cutFrames(options, [])
        self.assertEqualVector(result, [])

    def testOne(self):
       options = {"frameSize" : 100,
                  "hopSize" : 60,
                  "startFromZero" : True}
       found = self.cutFrames(options, [23])

       self.assertEqual(found[0][0], [23])
       self.assertEqualVector(found[0], [23]+[0]*(options['frameSize']-1))


    def testOneCentered(self):
        options = {"frameSize": 100,
                   "hopSize": 60,
                   "startFromZero": False}

        found = self.cutFrames(options, [23])
        expected = zeros(100)
        expected[50] = 23
        self.assertEqualVector(found[0], expected)

    def testLastFrame(self):
        options = {"frameSize": 100,
                   "hopSize": 60,
                   "startFromZero": True}

        found = self.cutFrames(options)
        expected = [range(100)]
        self.assertEqualMatrix(found, expected)

    def testLastFrame2(self):
        options = {"frameSize": 101,
                   "hopSize": 60,
                   "startFromZero": True}

        found = self.cutFrames(options)
        expected = [range(100)+[0]]
        self.assertEqualMatrix(found, expected)

    def testLastFrameCentered(self):
        options = {"frameSize": 100,
                   "hopSize": 60,
                   "startFromZero": False}

        found = self.cutFrames(options)
        expected = [[0]*50 + range(50),
                    range(10,100) + [0]*10,
                    range(70,100) + [0]*70]
        self.assertEqualMatrix(found, expected)


    def testLastFrameCentered2(self):
        options = {"frameSize": 102,
                   "hopSize": 60,
                   "startFromZero": False}

        found = self.cutFrames(options)
        expected = [[0]*51 + range(51),
                    range(9,100) + [0]*11,
                    range(69,100) + [0]*71]
        self.assertEqualMatrix(found, expected)


    def testBigHopSize(self):
        options = {"frameSize": 20,
                   "hopSize": 100,
                   "startFromZero": True}

        found = self.cutFrames(options)
        expected = [range(20)]
        self.assertEqualMatrix(found, expected)

    def testBigHopSize2(self):
        options = {"frameSize": 20,
                   "hopSize": 99,
                   "startFromZero": True}

        found = self.cutFrames(options)
        expected = [ range(20), [99] + [0]*19 ]
        self.assertEqualMatrix(found, expected)


    def testBigHopSizeCentered(self):
        options = {"frameSize": 20,
                   "hopSize": 100,
                   "startFromZero": False}

        found = self.cutFrames(options)
        expected = [[0]*10 + range(10),range(90,100) + [0]*10]
        self.assertEqualMatrix(found, expected)


    def testComplex(self):
        options = {"frameSize": 3,
                   "hopSize": 2,
                   "startFromZero": True}

        found = self.cutFrames(options, range(1, 6))
        expected = [[ 1, 2, 3 ],
                    [ 3, 4, 5 ]]
        self.assertEqualMatrix(found, expected)


    def testComplex2(self):
        options = {"frameSize": 2,
                   "hopSize": 1,
                   "startFromZero": True}

        found = self.cutFrames(options, range(1, 4))
        expected = [[ 1, 2 ],
                    [ 2, 3 ]]
        self.assertEqualMatrix(found, expected)


    def testComplexCentered(self):
        options = {"frameSize": 3,
                   "hopSize": 2,
                   "startFromZero": False}

        found = self.cutFrames(options, range(1, 6))
        expected = [[0,0,1], range(1,4), range(3,6),  [5,0,0]]
        self.assertEqualMatrix(found, expected)


    def testComplexCentered2(self):
        options = {"frameSize": 2,
                   "hopSize": 1,
                   "startFromZero": False}

        found = self.cutFrames(options, range(1, 4))
        expected = [[0,1], range(1,3), range(2,4), range(3,4)+[0]]
        self.assertEqualMatrix(found, expected)

    def testIncompatibleParams(self):
        self.assertConfigureFails(es.FrameCutter(), {'startFromZero' : False,
                                                     'validFrameThresholdRatio' : .6})

    def testDropLastFrame_StartFromZeroEvenFrameSize(self):
        expected = [range(1,  11),
                    range(11, 21),
                    range(21, 31),
                    range(31, 41),
                    range(41, 51)]

        # test with ratio = 1
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = self.cutFrames(options, range(1, 60))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .9
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': .9}
        found = self.cutFrames(options, range(1, 59))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 52))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_StartFromZeroEvenFrameSize(self):
        # test with ratio = 1
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = self.cutFrames(options, range(1, 61))
        expected = [range(1,  11),
                    range(11, 21),
                    range(21, 31),
                    range(31, 41),
                    range(41, 51),
                    range(51, 61)]
        self.assertEqualMatrix(found, expected)


        # test with ratio = .9
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': .9}
        found = self.cutFrames(options, range(1, 60))
        expected = [range(1,  11),
                    range(11, 21),
                    range(21, 31),
                    range(31, 41),
                    range(41, 51),
                    range(51, 60)+[0]]
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 53))
        expected = [range(1,  11),
                    range(11, 21),
                    range(21, 31),
                    range(31, 41),
                    range(41, 51),
                    range(51, 53)+[0]*8]
        self.assertEqualMatrix(found, expected)

    def testDropLastFrame_StartFromZeroOddFrameSize(self):
        expected = [range(1,  12),
                    range(12, 23),
                    range(23, 34),
                    range(34, 45),
                    range(45, 56)]

        # test with ratio = 1
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = self.cutFrames(options, range(1, 66))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .9
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': .9}
        found = self.cutFrames(options, range(1, 65))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 57))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_StartFromZeroOddFrameSize(self):
        # test with ratio = 1
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = self.cutFrames(options, range(1, 67))
        expected = [range(1,  12),
                    range(12, 23),
                    range(23, 34),
                    range(34, 45),
                    range(45, 56),
                    range(56, 67)]
        self.assertEqualMatrix(found, expected)


        # test with ratio = .9
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': .9}
        found = self.cutFrames(options, range(1, 66))
        expected = [range(1,  12),
                    range(12, 23),
                    range(23, 34),
                    range(34, 45),
                    range(45, 56),
                    range(56, 66)+[0]]
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 59))
        expected = [range(1,  12),
                    range(12, 23),
                    range(23, 34),
                    range(34, 45),
                    range(45, 56),
                    range(56, 59)+[0]*8]
        self.assertEqualMatrix(found, expected)

    def testDropLastFrame_EvenFrameSize(self):
        expected = [[0]*5+range(1, 6),
                          range(6, 16),
                          range(16, 26),
                          range(26, 36),
                          range(36, 46),
                          range(46, 56)]

        # test with ratio = .5 (highest threshold possible with
        # startFromZero == false)
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .5}
        found = self.cutFrames(options, range(1, 60))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 57))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_EvenFrameSize(self):
        # test with ratio = .5 (highest threshold possible with
        # startFromZero == false)
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .5}
        found = self.cutFrames(options, range(1, 61))
        expected = [[0]*5+range(1, 6),
                          range(6, 16),
                          range(16, 26),
                          range(26, 36),
                          range(36, 46),
                          range(46, 56),
                          range(56, 61)+[0]*5]
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 58))
        expected = [[0]*5+range(1, 6),
                          range(6, 16),
                          range(16, 26),
                          range(26, 36),
                          range(36, 46),
                          range(46, 56),
                          range(56, 58)+[0]*8]
        self.assertEqualMatrix(found, expected)

    def testDropLastFrame_OddFrameSize(self):
        expected = [[0]*6+range(1,  6),
                          range(6,  17),
                          range(17, 28),
                          range(28, 39),
                          range(39, 50)]

        # test with ratio = .5 (highest threshold possible with
        # startFromZero == false)
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': False, 'validFrameThresholdRatio': .5}
        found = self.cutFrames(options, range(1, 55))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 51))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_OddFrameSize(self):
        # test with ratio = .5 (highest threshold possible with
        # startFromZero == false)
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': False, 'validFrameThresholdRatio': .5}
        found = self.cutFrames(options, range(1, 56))
        expected = [[0]*6+range( 1,  6),
                          range( 6, 17),
                          range(17, 28),
                          range(28, 39),
                          range(39, 50),
                          range(50, 56)+[0]*5]
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = self.cutFrames(options, range(1, 53))
        expected = [[0]*6+range( 1,  6),
                          range( 6, 17),
                          range(17, 28),
                          range(28, 39),
                          range(39, 50),
                          range(50, 53)+[0]*8]
        self.assertEqualMatrix(found, expected)

    def cutAudioFile(self, filename, frameSize, hopSize, startFromZero, expectedNumFrames):
        loader = es.MonoLoader(filename=join(testdata.audio_dir, 'generated','synthesised', 'shortfiles', filename))
        fc = es.FrameCutter(frameSize=frameSize,
                         hopSize = hopSize,
                         startFromZero = startFromZero)
        p = Pool()
        loader.audio >> fc.signal
        fc.frame >> (p, 'audio.frames')
        run(loader)

        self.assertEqual(len(p['audio.frames']), expectedNumFrames)

    def testShortAudioFilesNormalHopSize(self):
        self.cutAudioFile("1024_samples.wav", 512, 256, True, 3)
        self.cutAudioFile("1989_samples.wav", 512, 256, True, 7)
        self.cutAudioFile("1024_samples.wav", 512, 256, False, 5)
        self.cutAudioFile("1989_samples.wav", 512, 256, False, 9)

    def testShortAudioFilesHopSizeLargerThanFrameSize(self):
        self.cutAudioFile("1024_samples.wav", 256, 512, True, 2)
        self.cutAudioFile("1989_samples.wav", 256, 512, True, 4)
        self.cutAudioFile("1024_samples.wav", 256, 512, False, 3)
        self.cutAudioFile("1989_samples.wav", 256, 512, False, 5)

    def testShortAudioFilesHopSizePastEndOfStream(self):
        self.cutAudioFile("1024_samples.wav", 512, 8192, True, 1)
        self.cutAudioFile("1989_samples.wav", 512, 8192, True, 1)
        self.cutAudioFile("1024_samples.wav", 512, 8192, False, 1)
        self.cutAudioFile("1989_samples.wav", 512, 8192, False, 1)

    def testSilentFrames(self):
        input = [0]*1024*3 # 3 frames of 1024
        gen = VectorInput(input)
        pool = Pool()
        expectedFrames = 5

        # adding noise
        frameCutter = es.FrameCutter(frameSize = 1024,
                                     hopSize = 512,
                                     startFromZero = True,
                                     silentFrames = "noise")

        gen.data >> frameCutter.signal
        frameCutter.frame >> (pool, 'frames')
        run(gen)
        self.assertEqual(len(pool['frames']), expectedFrames)
        energy = std.Energy()
        for f in pool['frames']:
            self.assertTrue(essentia._essentia.isSilent(f))
            self.assertTrue(energy(f) != 0)

        pool.remove('frames')
        reset(gen)

        # keep silent frames
        frameCutter.configure(frameSize = 1024,
                              hopSize = 512,
                              startFromZero = True,
                              silentFrames = "keep")

        run(gen)

        self.assertEqual(len(pool['frames']), expectedFrames)
        energy = std.Energy()
        for f in pool['frames']:
            self.assertTrue(essentia._essentia.isSilent(f))
            self.assertTrue(energy(f) == 0)

        pool.remove('frames')
        reset(gen)

        # drop silent frames
        frameCutter.configure(frameSize = 1024,
                              hopSize = 512,
                              startFromZero = True,
                              silentFrames = "drop")

        run(gen)
        self.assertTrue(len(pool.descriptorNames())==0)



suite = allTests(TestFrameCutter_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
