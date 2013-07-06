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
from essentia.streaming import ChordsDescriptors as sChordsDescriptors

#tip:
#circleOfFifth : "C", "Em", "G", "Bm", "D", "F#m", "A", "C#m", "E", "G#m", "B", "D#m", "F#", "A#m", "C#", "Fm", "G#", "Cm", "D#", "Gm", "A#", "Dm", "F", "Am";

class TestChordsDescriptors(TestCase):
      def testRegressionC(self):
          chords = ['C', 'F', 'C', 'G', 'C', 'Am', 'Dm', 'G']
          result = ChordsDescriptors()(chords, 'C', 'major')
          expectedHist = [3./8., 0, 2./8., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                          0, 0, 0, 0, 0, 0, 1./8., 1./8., 1./8.]
          self.assertEqualVector(result[0], [hist*100 for hist in expectedHist])
          self.assertEqual(result[1], 5.0/8.0) #chord rate
          self.assertEqual(result[2], 7.0/8.0) #change rate
          self.assertEqual(result[3], 'C')     #key
          self.assertEqual(result[4], 'major') #scale

      def testRegressionG(self):
          # same test as above, but input key is different which will affect
          # the output of the histogram:
          chords = ['C', 'F', 'C', 'G', 'C', 'Am', 'Dm', 'G']
          result = ChordsDescriptors()(chords, 'G', 'major')
          expectedHist = [2./8., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                          0, 0, 0, 1./8., 1./8., 1./8., 3./8., 0]

          self.assertEqualVector(result[0], [hist*100 for hist in expectedHist])
          self.assertEqual(result[1], 5.0/8.0) #chord rate
          self.assertEqual(result[2], 7.0/8.0) #change rate
          self.assertEqual(result[3], 'C')     #key
          self.assertEqual(result[4], 'major') #scale

      def testEqualProb(self):
          chords = [ 'C', 'G', 'C', 'G', 'C', 'G' ]
          result = ChordsDescriptors()(chords, 'C', 'major')
          expectedHist = [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0]

          self.assertEqualVector(result[0], [hist*100 for hist in expectedHist])
          self.assertAlmostEqual(result[1], 2.0/6.0) #chord rate
          self.assertAlmostEqual(result[2], 5.0/6.0) #change rate
          self.assertEqual(result[3], 'C')     #key
          self.assertEqual(result[4], 'major') #scale

      def testCaseSensitivity(self):
          chords = [ 'C', 'G', 'C', 'G', 'C', 'G' ]
          result = ChordsDescriptors()(chords, 'c', 'MaJoR')
          expectedHist = [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0]

          self.assertEqualVector(result[0], [hist*100 for hist in expectedHist])
          self.assertAlmostEqual(result[1], 2.0/6.0) #chord rate
          self.assertAlmostEqual(result[2], 5.0/6.0) #change rate
          self.assertEqual(result[3], 'C')     #key
          self.assertEqual(result[4], 'major') #scale

      def testUnknownChord(self):
          chords = [ 'Cb', 'G', 'C', 'G', 'C', 'G' ] # Cb will be ignored
          result = ChordsDescriptors()(chords, 'C', 'major')
          expectedHist = [2./6., 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0]

          self.assertAlmostEqualVector(result[0], [hist*100 for hist in expectedHist])
          self.assertAlmostEqual(result[1], 2.0/6.0) #chord rate
          self.assertAlmostEqual(result[2], 5.0/6.0) #change rate
          self.assertEqual(result[3], 'G')     #key
          self.assertEqual(result[4], 'major') #scale

      def testUnknownKey(self):
          self.assertComputeFails(ChordsDescriptors(), ['C', 'C'], 'Cb', 'major')

      def testEmpty(self):
          self.assertComputeFails(ChordsDescriptors(), [], 'C', 'major')

      def testOne(self):
          result = ChordsDescriptors()(['C'], 'C', 'major')
          self.assertEqualVector(result[0], [100]+[0]*23)
          self.assertEqual(result[1], 1.0) #chord rate
          self.assertEqual(result[2], 0.0) #change rate
          self.assertEqual(result[3], 'C')     #key
          self.assertEqual(result[4], 'major') #scale

      def testRegressionStreaming(self):
          chords = ['C', 'F', 'C', 'G', 'C', 'Am', 'Dm', 'G']

          chordsGen = VectorInput(chords)
          keyGen = VectorInput(['C'])
          scaleGen = VectorInput(['major'])

          algo = sChordsDescriptors()
          pool = Pool()

          chordsGen.data >> algo.chords
          keyGen.data    >> algo.key
          scaleGen.data   >> algo.scale
          algo.chordsHistogram >>   (pool, "hist")
          algo.chordsNumberRate >>  (pool, "rate")
          algo.chordsChangesRate >> (pool, "change")
          algo.chordsKey >>         (pool, "key")
          algo.chordsScale >>       (pool, "scale")


          keyGen.push("data", 'C')
          scaleGen.push("data", 'major')
          run(chordsGen)

          expectedHist = [3./8., 0, 2./8., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                          0, 0, 0, 0, 0, 0, 1./8., 1./8., 1./8.]
          self.assertEqualVector(pool['hist'], [hist*100 for hist in expectedHist])
          self.assertEqual(pool['rate'], 5.0/8.0) #chord rate
          self.assertEqual(pool['change'], 7.0/8.0) #change rate
          self.assertEqual(pool['key'], 'C')     #key
          self.assertEqual(pool['scale'], 'major') #scale



suite = allTests(TestChordsDescriptors)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
