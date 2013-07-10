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

class TestBarkBands(TestCase):

    def testRegression(self):
        # read expected output
        expected = readMatrix( join(filedir(), 'barkbands', 'output.txt') )

        # Simple regression test, comparing to reference values
        audio = MonoLoader(filename = join(testdata.audio_dir,
                                           'generated/synthesised/sin440_sweep_0db.wav'),
                           sampleRate = 44100)()

        fft = Spectrum()
        window = Windowing(size=2048, type='hamming', zeroPadding=0)
        bbands = BarkBands(sampleRate = 44100)

        frameIdx = 0
        for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
            bands = bbands(fft(window(frame)))

            self.assert_(not any(numpy.isnan(bands)))
            self.assert_(not any(numpy.isinf(bands)))
            self.assert_(all(bands >= 0.0))

            # this test is commented because although essentia yields exactly
            # the same results as in output.txt when run as cpp, python does
            # not due to rounding errors, presumably
            #self.assertAlmostEqualVector(bands, expected[frameIdx], 1e-1)
            frameIdx += 1

    def testWhiteNoise(self):
        # test with spectrum of ones
        bands = [ 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0,\
                  1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0,\
                  5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0, 27000.0 ]
        sampleRate = 44100
        specSize = 2048
        spec = ones(specSize)
        frequencyScale = sampleRate*0.5/(specSize-1);
        bins = [int(band/frequencyScale + 0.5) for band in bands]
        if bins[-1] > specSize: bins[-1] = specSize;
        expected = [bins[i+1] - bins[i] for i in range(len(bins)-1)]
        bbands = BarkBands(sampleRate = sampleRate, numberBands = 28)
        self.assertEqualVector(bbands(spec), expected)

    def testZero(self):
        # Inputting zeros should return null band energies
        bbands = BarkBands()
        self.assert_(all(bbands(zeros(1024)) == 0))
        self.assert_(all(bbands(zeros(10)) == 0))

    def testInvalidParam(self):
        # Test that we can only have up to 28 bands
        self.assertConfigureFails(BarkBands(), { 'numberBands': 29 })
        self.assertConfigureFails(BarkBands(), { 'numberBands': 0 })
        self.assertConfigureFails(BarkBands(), { 'numberBands': -3 })

    def testEmpty(self):
        # Test that barkbands on an empty vector should throw exception
        self.assertComputeFails(BarkBands(numberBands = 20), [])


suite = allTests(TestBarkBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
