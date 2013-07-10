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

class TestPeakDetection(TestCase):

    def testPeakAtBegining(self):
        input=[1,0,0,0]
        inputSize = len(input)
        config = { 'range': inputSize -1,  'maxPosition': inputSize-1,  'minPosition': 0,  'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [0])
        self.assertEqualVector(vals, [1])

    def testPeakAtMinPosition(self):
        peakPos = 3
        input=[0,0,0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize -1,  'maxPosition': inputSize-1, 'minPosition': peakPos,  'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakZero(self):
        peakPos = 0
        input=[0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize-1,  'maxPosition': inputSize-1, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakOne(self):
        peakPos = 1
        input=[0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize-1,  'maxPosition': inputSize-1, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakTwo(self):
        peakPos = 2
        input=[0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize-1,  'maxPosition': inputSize-1, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakThree(self):
        peakPos = 3
        input=[0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize-1,  'maxPosition': inputSize-1, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakFour(self):
        peakPos = 4
        input=[0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize-1,  'maxPosition': inputSize-1, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakBeforeEnd(self):
        peakPos = 3
        input=[0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize-1,  'maxPosition': inputSize-1, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakAtMaxPosition(self):
        peakPos = 3
        input=[0,0,0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize -1,  'maxPosition': peakPos, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testPeakEnd(self):
        input=[0,0,0,0,0,0,0]
        inputSize = len(input)
        peakPos = inputSize-1
        input[peakPos] = 1
        config = { 'range': inputSize -1,  'maxPosition': peakPos, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testMaxPositionLargerThanSize(self):
        peakPos = 3
        input=[0,0,0,0,0,0,0]
        input[peakPos] = 1
        inputSize = len(input)
        config = { 'range': inputSize -1,  'maxPosition': 10*peakPos, 'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [peakPos])
        self.assertEqualVector(vals, [1])

    def testRegression(self):
        inputSize = 1024

        config = { 'range': inputSize -1,  'maxPosition': inputSize,  'minPosition': 0,  'orderBy': 'amplitude' }
        pdetect = PeakDetection(**config)

        pos1 = 3
        val1 = 1.0
        inputOne = [0] * inputSize
        inputOne[pos1] = val1
        (posis, vals) = pdetect(inputOne)
        self.assertEqualVector(posis, [pos1])
        self.assertEqualVector(vals, [val1])

        pos1 = 3
        val1 = 1.0
        pos2 = 512
        val2 = 3.0
        inputTwo = [0] * inputSize
        inputTwo[pos1] = val1
        inputTwo[pos2] = val2
        (posis, vals) = pdetect(inputTwo)
        self.assertEqualVector(posis, [pos2, pos1])
        self.assertEqualVector(vals, [val2, val1])

        config['orderBy'] = 'position'
        pdetect.configure(**config)
        (posis, vals) = pdetect(inputTwo)
        self.assertEqualVector(posis, [pos1, pos2])
        self.assertEqualVector(vals, [val1, val2])

    def testPlateau(self):
        inputSize = 1024
        pos = range(3, 7)
        val = 1.0
        input = [0] * inputSize
        for i in range(len(pos)):
            input[pos[i]] = val

        # with interpolation:
        config = { 'range': inputSize -1,  'maxPosition': inputSize, 'minPosition': 0,  'orderBy': 'amplitude', 'interpolate' : True }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [0.5*(pos[0]+pos[-1])])
        self.assertEqualVector(vals, [val])

        # no interpolation:
        config = { 'range': inputSize -1,  'maxPosition': inputSize, 'minPosition': 0,  'orderBy': 'amplitude', 'interpolate' : False }
        pdetect = PeakDetection(**config)

        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [pos[0]])
        self.assertEqualVector(vals, [val])

    def testStairCase(self):
        # should find the first postition of the last step
        inputSize = 1024
        pos = [range(3, 7), range(7,14), range(14,20)]
        val = [1.0, 2.0, 3.0]
        input = [0] * inputSize
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                input[pos[i][j]] = val[i]

        # no interpolation:
        config = { 'range': inputSize -1,  'maxPosition': inputSize, 'minPosition': 0,  'orderBy': 'amplitude', 'interpolate' : False }
        pdetect = PeakDetection(**config)
        (posis, vals) = pdetect(input)
        self.assertEqualVector(posis, [pos[-1][0]])
        self.assertEqualVector(vals, [val[-1]])


    def testZero(self):
        inputSize = 1024
        input = [0] * inputSize
        pdetect = PeakDetection()
        (posis, vals) = pdetect(input)
        self.assert_(len(posis) == 0)
        self.assert_(len(vals) == 0)

    def testEmpty(self):
        # Feeding an empty array shouldn't crash and throw an exception
        self.assertComputeFails(PeakDetection(),  [])

    def testOne(self):
        # Feeding an array of size 1 shouldn't crash and throw an exception
        self.assertComputeFails(PeakDetection(), [0])

    def testInvalidParam(self):
        self.assertConfigureFails(PeakDetection(), {'range': 0})
        self.assertConfigureFails(PeakDetection(), {'maxPeaks': 0})
        self.assertConfigureFails(PeakDetection(), {'maxPosition': 0})
        self.assertConfigureFails(PeakDetection(), {'minPosition': -1})
        PeakDetection(threshold=0)
        self.assertConfigureFails(PeakDetection(), {'minPosition': 1.01,  'maxPosition': 1})

suite = allTests(TestPeakDetection)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
