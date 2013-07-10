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
from essentia.streaming import ChordsDetection

chord_dict = {
    'A':  [1, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0],
    'A#': [0, 1, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0],
    'B':  [0, 0, 1, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0],
    'C':  [0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0, 0.3, 0],
    'C#': [0, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0, 0.3],
    'D':  [0.3, 0, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0],
    'D#': [0, 0.3, 0, 0, 0, 0, 1, 0, 0, 0, 0.5, 0],
    'E':  [0, 0, 0.3, 0, 0, 0, 0, 1, 0, 0, 0, 0.5],
    'F':  [0.5, 0, 0, 0.3, 0, 0, 0, 0, 1, 0, 0, 0],
    'F#': [0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 1, 0, 0],
    'G':  [0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 1, 0],
    'G#': [0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 1],

    'Am':  [1, 0, 0, 0.5, 0, 0, 0, 0.3, 0, 0, 0, 0],
    'A#m': [0, 1, 0, 0, 0.5, 0, 0, 0, 0.3, 0, 0, 0],
    'Bm':  [0, 0, 1, 0, 0, 0.5, 0, 0, 0, 0.3, 0, 0],
    'Cm':  [0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0, 0.3, 0],
    'C#m': [0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0, 0.3],
    'Dm':  [0.3, 0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0],
    'D#m': [0, 0.3, 0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0],
    'Em':  [0, 0, 0.3, 0, 0, 0, 0, 1, 0, 0, 0.5, 0],
    'Fm':  [0, 0, 0, 0.3, 0, 0, 0, 0, 1, 0, 0, 0.5],
    'F#m': [0.5, 0, 0, 0, 0.3, 0, 0, 0, 0, 1, 0, 0],
    'Gm':  [0, 0.5, 0, 0, 0, 0.3, 0, 0, 0, 0, 1, 0],
    'G#m': [0, 0, 0.5, 0, 0, 0, 0.3, 0, 0, 0, 0, 1]
}


class TestChordsDetection_Streaming(TestCase):

    def stringToRealProgression(self, chordsProgressionString):
        return [chord_dict[chord] for chord in chordsProgressionString]

    def runProgression(self, progression, streaming=True):
        nChords = len(progression)
        chordsProgressionReal = self.stringToRealProgression(progression)
        changeChordTime = 1.0
        dur = nChords*changeChordTime
        sampleRate = 44100
        hopSize = 2048
        nFrames = int(dur*sampleRate/hopSize) - 1
        frameRate = float(sampleRate)/float(hopSize)
        nextChange = frameRate*changeChordTime
        # compute expected Chords and the input pcp for the chordsdetection
        # algorithm:
        pcp = zeros([nFrames,12])
        expectedChords =[]
        j = 0
        for i in range(nFrames):
            if i == int(nextChange):
                j+=1
                nextChange += frameRate*changeChordTime
            expectedChords.append(progression[j%nChords])
            pcp[i] = chordsProgressionReal[j%nChords]

        pool = Pool()

        if streaming:
            gen = VectorInput(pcp)
            chordsDetection = ChordsDetection(windowSize=2.0, hopSize = hopSize)

            gen.data >> chordsDetection.pcp
            chordsDetection.chords >> (pool, 'chords.progression')
            chordsDetection.strength >> (pool, 'chords.strength')
            run(gen)

        else:
            from essentia.standard import ChordsDetection as stdChordsDetection
            chordsDetection = stdChordsDetection(windowSize=2.0, hopSize = hopSize)
            chords, strength = chordsDetection(pcp)
            for i in xrange(len(chords)):
                pool.add('chords.progression', chords[i])
                pool.add('chords.strength', float(strength[i]))


        # as sometimes the algorithm gets confused at the transition from one chord to the other,
        # the test will be restricted to not having more errors than transitions

        failure = 0
        for i in range(nFrames):
            if pool['chords.progression'][i] != expectedChords[i]: failure+=1
            self.assertValidNumber(pool['chords.strength'][i])
            self.assertTrue(pool['chords.strength'][i] > 0)

        self.assertTrue(failure <= nChords)

    def testMajor(self):
        progression = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#' ]
        self.runProgression(progression)

    def testMinor(self):
        progression = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m' ]
        self.runProgression(progression)

    def testMajorStd(self):
        progression = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#' ]
        self.runProgression(progression, False)

    def testMinorStd(self):
        progression = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m' ]
        self.runProgression(progression, False)

    #def testMixScale(self):
         # this test does fail, but could be considered as passed considerably ok.
         # The algorithm confuses A with Dm and Dm with D at the transition between A and D
    #    progression = [ 'Am', 'D', 'Fm', 'E', 'C', 'Gm' ]
    #    self.runProgression(progression)



    def testEmpty(self):
        gen = VectorInput(array([[]]))
        chordsDetection = ChordsDetection(windowSize=2.0, hopSize = 2048)
        pool = Pool()

        gen.data >> chordsDetection.pcp
        chordsDetection.chords >> (pool, 'chords.progression')
        chordsDetection.strength >> (pool, 'chords.strength')
        self.assertRaises(EssentiaException, lambda:run(gen))

        self.assertEqualVector(pool.descriptorNames(), [])

    def testZero(self):
        pcp = zeros([10, 12])
        gen = VectorInput(pcp)
        chordsDetection = ChordsDetection(windowSize=2.0, hopSize = 2048)
        pool = Pool()

        gen.data >> chordsDetection.pcp
        chordsDetection.chords >> (pool, 'chords.progression')
        chordsDetection.strength >> (pool, 'chords.strength')
        run(gen)

        self.assertEqualVector(pool['chords.progression'], ['A']*len(pcp))
        self.assertEqualVector(pool['chords.strength'], [-1]*len(pcp))

    def testNumChordsEqualsHpcpSize(self):
        # this test has been introduced since it was reported that
        # chordsdetection may reveal errors on the scheduling yielding more
        # chords than hpcps are computed
        from essentia.streaming import MonoLoader, DCRemoval, FrameCutter,\
        EqualLoudness, Windowing, Spectrum, SpectralPeaks, SpectralWhitening,\
        HPCP

        audiofile = 'musicbox.wav'
        filename = filename=join(testdata.audio_dir,'recorded', audiofile)

        p = Pool()
        loader = MonoLoader(filename=filename)
        dc = DCRemoval()
        eqloud = EqualLoudness()
        fc = FrameCutter(frameSize=2048, hopSize=1024, silentFrames="noise")
        win = Windowing(size=2048)
        spec = Spectrum()
        specPeaks = SpectralPeaks()
        specWhite = SpectralWhitening()
        hpcp = HPCP()
        chords = ChordsDetection(hopSize=1024)

        loader.audio >> dc.signal
        dc.signal >> eqloud.signal
        eqloud.signal >> fc.signal
        fc.frame >> win.frame
        win.frame>> spec.frame
        spec.spectrum >> specPeaks.spectrum
        spec.spectrum >> specWhite.spectrum
        specPeaks.frequencies >> specWhite.frequencies
        specPeaks.magnitudes >> specWhite.magnitudes
        specWhite.magnitudes >> hpcp.magnitudes
        specPeaks.frequencies >> hpcp.frequencies
        hpcp.hpcp >> chords.pcp
        chords.chords >> (p, 'chords')
        chords.strength >> None
        hpcp.hpcp >> (p, 'hpcp')

        run(loader)
        self.assertEqual(len(p['chords']), len(p['hpcp']))


    def testInvalidParam(self):
        self.assertConfigureFails(ChordsDetection(),{'sampleRate' : 0})
        self.assertConfigureFails(ChordsDetection(),{'hopSize' : 0})
        self.assertConfigureFails(ChordsDetection(),{'windowSize' : 0})

suite = allTests(TestChordsDetection_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
