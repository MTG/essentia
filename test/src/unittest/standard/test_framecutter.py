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


def cutFrames(params, input = range(100)):
    if not 'validFrameThresholdRatio' in params:
        params['validFrameThresholdRatio'] = 0
    framegen = FrameGenerator(input,
                              frameSize = params['frameSize'],
                              hopSize = params['hopSize'],
                              validFrameThresholdRatio = params['validFrameThresholdRatio'],
                              startFromZero = params['startFromZero'])

    return [ frame for frame in framegen ]



class TestFrameCutter(TestCase):

    def testInvalidParam(self):
        fcreator = FrameCutter()
        self.assertConfigureFails(fcreator, { 'frameSize': 0 })
        self.assertConfigureFails(fcreator, { 'frameSize': -23 })
        self.assertConfigureFails(fcreator, { 'hopSize': 0 })
        self.assertConfigureFails(fcreator, { 'hopSize': -23 })

    # extreme conditions
    def testEmpty(self):
        found = cutFrames({ 'frameSize': 100, 'hopSize': 60, 'startFromZero': True }, array([]))
        expected = []
        self.assertEqualMatrix(found, expected)

    def testEmptyCentered(self):
        found = cutFrames({ 'frameSize': 100, 'hopSize': 60, 'startFromZero': False }, array([]))
        expected = []
        self.assertEqualMatrix(found, expected)

    def testOne(self):
        found = cutFrames({ 'frameSize': 100, 'hopSize': 60, 'startFromZero': True }, array([ 23 ]))
        expected = [ [23] + [0]*99 ]
        self.assertEqualMatrix(found, expected)

    def testOneCentered(self):
        found = cutFrames({ 'frameSize': 100, 'hopSize': 60, 'startFromZero': False }, array([ 23 ]))
        expected = [ [0]*50 + [23] + [0]*49 ]
        self.assertEqualMatrix(found, expected)

    # test we get the last frame right
    def testLastFrame(self):
        found = cutFrames({ 'frameSize': 100, 'hopSize': 60, 'startFromZero': True })
        expected = [range(100)]
        self.assertEqualMatrix(found, expected)

    def testLastFrame2(self):
        found = cutFrames({ 'frameSize': 101, 'hopSize': 60, 'startFromZero': True })
        expected = [ range(100)+[0] ]
        self.assertEqualMatrix(found, expected)

    def testLastFrameCentered(self):
        found = cutFrames({ 'frameSize': 100, 'hopSize': 60, 'startFromZero': False })
        expected = [ [0]*50 + range(50),
                     range(10, 100) + [0]*10,
                     range(70, 100) + [0]*70 ]
        self.assertEqualMatrix(found, expected)

    def testLastFrameCentered2(self):
        found = cutFrames({ 'frameSize': 102, 'hopSize': 60, 'startFromZero': False })
        expected = [ [0]*51 + range(51),
                     range(9, 100) + [0]*11,
                     range(69, 100) + [0]*71 ]
        self.assertEqualMatrix(found, expected)

    def testLastFrameCentered3(self):
        found = cutFrames({ 'frameSize': 101, 'hopSize': 60, 'startFromZero': False })
        expected = [ [0]*51 + range(50),
                     range(9, 100) + [0]*10,
                     range(69, 100) + [0]*70 ]
        self.assertEqualMatrix(found, expected)

    # test hopSize > frameSize
    def testBigHopSize(self):
        found = cutFrames({ 'frameSize': 20, 'hopSize': 100, 'startFromZero': True })
        expected = [ range(20) ]
        self.assertEqualMatrix(found, expected)

    def testBigHopSize2(self):
        found = cutFrames({ 'frameSize': 20, 'hopSize': 40, 'startFromZero': True })
        expected = [ range(20),  range(40,60), range(80,100)]
        self.assertEqualMatrix(found, expected)

    def testBigHopSizeCentered(self):
        found = cutFrames({ 'frameSize': 20, 'hopSize': 100, 'startFromZero': False })
        expected = [ [0]*10 + range(10),
                     range(90,100) + [0]*10 ]
        self.assertEqualMatrix(found, expected)

    def testBigHopSizeCentered2(self):
        found = cutFrames({ 'frameSize': 20, 'hopSize': 40, 'startFromZero': False})
        expected = [ [0]*10+range(10),  range(30,50), range(70,90)]
        self.assertEqualMatrix(found, expected)

    # full-fledged tricky tests
    def testComplex(self):
        found = cutFrames({ 'frameSize': 3, 'hopSize': 2, 'startFromZero': True }, range(1, 6))
        expected = [[ 1, 2, 3 ],
                    [ 3, 4, 5 ]]
        self.assertEqualMatrix(found, expected)

    def testComplex2(self):
        found = cutFrames({ 'frameSize': 2, 'hopSize': 1, 'startFromZero': True }, range(1, 4))
        expected = [[ 1, 2 ],
                    [ 2, 3 ]]
        self.assertEqualMatrix(found, expected)

    def testComplexCentered(self):
        found = cutFrames({ 'frameSize': 3, 'hopSize': 2, 'startFromZero': False }, range(1, 6))
        expected = [[ 0, 0, 1 ],
                    [ 1, 2, 3 ],
                    [ 3, 4, 5 ],
                    [ 5, 0, 0 ]]
        self.assertEqualMatrix(found, expected)

    def testEOF(self):
        found = cutFrames({ 'frameSize': 3, 'hopSize': 2, 'startFromZero': True, 'lastFrameToEndOfFile':True }, range(1, 6))
        expected = [[ 1, 2, 3 ],
                    [ 3, 4, 5 ]]
        self.assertEqualMatrix(found, expected)

    def testEOF2(self):
        found = cutFrames({ 'frameSize': 2, 'hopSize': 1, 'startFromZero': True, 'lastFrameToEndOfFile':True }, range(1, 4))
        expected = [[ 1, 2 ],
                    [ 2, 3 ]]
        self.assertEqualMatrix(found, expected)

    def testEOF3(self):
        found = cutFrames({ 'frameSize': 2, 'hopSize': 3, 'startFromZero': True, 'lastFrameToEndOfFile':True }, range(1, 6))
        expected = [[ 1, 2 ],
                    [ 4, 5 ]]
        self.assertEqualMatrix(found, expected)

        found = cutFrames({ 'frameSize': 2, 'hopSize': 3, 'startFromZero': True, 'lastFrameToEndOfFile':True }, range(1, 8))
        expected = [[ 1, 2 ],
                    [ 4, 5 ],
                    [ 7, 0]]
        self.assertEqualMatrix(found, expected)

        found = cutFrames({ 'frameSize': 4, 'hopSize': 2, 'startFromZero': True, 'lastFrameToEndOfFile':True }, range(1, 8))
        expected = [[ 1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 0]]
        self.assertEqualMatrix(found, expected)

    def testComplexCentered2(self):
        found = cutFrames({ 'frameSize': 2, 'hopSize': 1, 'startFromZero': False }, range(1, 4))
        expected = [[ 0, 1 ],
                    [ 1, 2 ],
                    [ 2, 3 ],
                    [ 3, 0 ]]
        self.assertEqualMatrix(found, expected)

    def testDropLastFrame_StartFromZeroEvenFrameSize(self):
        expected = [range(1,  11),
                    range(11, 21),
                    range(21, 31),
                    range(31, 41),
                    range(41, 51)]

        # test with ratio = 1
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = cutFrames(options, range(1, 60))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .9
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': .9}
        found = cutFrames(options, range(1, 59))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': .2}
        found = cutFrames(options, range(1, 52))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_StartFromZeroEvenFrameSize(self):
        # test with ratio = 1
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = cutFrames(options, range(1, 61))
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
        found = cutFrames(options, range(1, 60))
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
        found = cutFrames(options, range(1, 53))
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
        found = cutFrames(options, range(1, 66))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .9
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': .9}
        found = cutFrames(options, range(1, 65))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': .2}
        found = cutFrames(options, range(1, 57))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_StartFromZeroOddFrameSize(self):
        # test with ratio = 1
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': True, 'validFrameThresholdRatio': 1}
        found = cutFrames(options, range(1, 67))
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
        found = cutFrames(options, range(1, 66))
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
        found = cutFrames(options, range(1, 59))
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
        found = cutFrames(options, range(1, 60))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = cutFrames(options, range(1, 57))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_EvenFrameSize(self):
        # test with ratio = .5 (highest threshold possible with
        # startFromZero == false)
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .5}
        found = cutFrames(options, range(1, 61))
        expected = [[0]*5+range(1, 6),
                          range(6, 16),
                          range(16, 26),
                          range(26, 36),
                          range(36, 46),
                          range(46, 56),
                          range(56, 61)+[0]*5]
        self.assertEqualMatrix(found, expected)

        # ttestDontDropLastFrame_OddFrameSizeest with ratio = .2
        options = {'frameSize': 10, 'hopSize': 10,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = cutFrames(options, range(1, 58))
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
        found = cutFrames(options, range(1, 55))
        self.assertEqualMatrix(found, expected)

        # test with ratio = .2
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': False, 'validFrameThresholdRatio': .2}
        found = cutFrames(options, range(1, 51))
        self.assertEqualMatrix(found, expected)

    def testDontDropLastFrame_OddFrameSize(self):
        # test with ratio = .5 (highest threshold possible with
        # startFromZero == false)
        options = {'frameSize': 11, 'hopSize': 11,
                   'startFromZero': False, 'validFrameThresholdRatio': .5}
        found = cutFrames(options, range(1, 56))
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
        found = cutFrames(options, range(1, 53))
        expected = [[0]*6+range( 1,  6),
                          range( 6, 17),
                          range(17, 28),
                          range(28, 39),
                          range(39, 50),
                          range(50, 53)+[0]*8]
        self.assertEqualMatrix(found, expected)

    def cutAudioFile(self, filename, frameSize, hopSize, startFromZero, expectedNumFrames):
        audio = MonoLoader(filename=join(testdata.audio_dir,\
                           'generated','synthesised', 'shortfiles', filename))()
        options = {'frameSize': frameSize,
                   'hopSize': hopSize,
                   'startFromZero': startFromZero}

        found = cutFrames(options, audio)
        self.assertEqual(len(found), expectedNumFrames)

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



suite = allTests(TestFrameCutter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
