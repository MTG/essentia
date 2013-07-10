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


class TestTempoScaleBands(TestCase):

    def testRegression(self):
       # Testing that results are not inf nor nan, but real numbers
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_sweep_0db.wav'),
                           sampleRate = 44100)()

        fft = Spectrum()
        window = Windowing(type = 'hamming')
        nbands = [0, 100, 200, 300, 400, 500, 600, 700, 800]
        fbands = FrequencyBands(frequencyBands = nbands,
                                sampleRate = 44100)
        tempobands = TempoScaleBands()
        bandsgain = [2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5]

        for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
            scaledbands, cumul = tempobands(fbands(fft(window(frame))))

            self.assert_(not any(numpy.isnan(scaledbands)))
            self.assert_(not any(numpy.isinf(scaledbands)))
            self.assert_(all(scaledbands >= 0.0))

            self.assert_(not numpy.isnan(cumul))
            self.assert_(not numpy.isinf(cumul))
            self.assert_(cumul >= 0.0)

    def testConstantInput(self):
       # When input is constant should yiled zero
        bandsgain = [2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5]
        spectrum = [1]*len(bandsgain)
        tempoScale = TempoScaleBands(bandsGain=bandsgain)
        i = 0
        while (i<10):
            scaledbands, cumul = tempoScale(spectrum)
            i+=1

        self.assertEqualVector(scaledbands, zeros(len(bandsgain)))
        self.assertEqual(cumul, 0.0)

    def testZero(self):
        pass
        # Inputting zeros should return null band energies
        fbands = [0, 100, 200, 300, 400, 500, 600, 700, 800]
        scaledbands, cumul = TempoScaleBands()(FrequencyBands(frequencyBands=fbands)(zeros(1024)))
        self.assertEqualVector(scaledbands, zeros(len(fbands)-1))
        self.assertEqual(cumul, 0.0)

    def testInvalidParam(self):
        # Test that we must give valid frequency ranges or order
        self.assertConfigureFails(TempoScaleBands(),{'bandsGain': []})
        self.assertConfigureFails(TempoScaleBands(), { 'frameTime': -1 })
        self.assertComputeFails(TempoScaleBands(bandsGain=[1, 2, 1]), zeros(512))

    def testEmpty(self):
        # Test that FrequencyBands on an empty vector should return null band energies
         self.assertComputeFails(TempoScaleBands(), [])


suite = allTests(TestTempoScaleBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
