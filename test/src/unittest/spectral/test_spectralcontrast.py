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


class TestSpectralContrast(TestCase):

    def testRegression(self):
        # Simple regression test, comparing to reference values
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/musicbox.wav'),
                           sampleRate = 44100)()

        fft = Spectrum()
        window = Windowing(type = 'hamming')
        SC = SpectralContrast(sampleRate = 44100)

        expected = 0

        sc = []
        valleys = []
        for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
            result = SC(fft(window(frame)))
            self.assert_(not any(numpy.isnan(result[0])))
            self.assert_(not any(numpy.isinf(result[1])))
            sc += [result[0]]
            valleys += [result[1]]
        self.assertAlmostEqual(numpy.mean(sc), -0.604606057431, 1e-5)
        self.assertAlmostEqual(numpy.mean(valleys), -8.55062127501, 1e-5)


    def testZero(self):
        SC = SpectralContrast(sampleRate = 44100)
        sc, valleys = SC(zeros(1025))
        self.assertAlmostEqual(numpy.mean(sc), -1)
        self.assertAlmostEqual(numpy.mean(valleys), numpy.log(1e-30))

    def testOnes(self):
        SC = SpectralContrast(sampleRate = 44100)
        sc, valleys = SC(ones(1025))
        self.assertAlmostEqual(numpy.mean(sc), -1)
        self.assertAlmostEqual(numpy.mean(valleys), 0)

    def testConstant(self):
        SC = SpectralContrast(sampleRate = 44100)
        sc, valleys = SC([0.5]*1025)
        self.assertAlmostEqual(numpy.mean(sc), -1)
        self.assertAlmostEqual(numpy.mean(valleys),-0.6931471825, 1e-7)

    def testCompare(self):
        spec0 = [1]*1025
        spec1 = [1]*1015 + [0]*10
        spec2 = [1]*10 + [0]*1015
        sr = 44100
        SC = SpectralContrast(sampleRate = sr, highFrequencyBound = sr/2)

        sc0 = SC(spec0)
        sc1 = SC(spec1)
        sc2 = SC(spec2)
        self.assertTrue(numpy.mean(sc1[0]) < numpy.mean(sc2[0]))
        self.assertTrue(numpy.mean(sc0[0]) < numpy.mean(sc2[0]))
        self.assertTrue(numpy.mean(sc0[0]) < numpy.mean(sc1[0]))

    def testInvalidParam(self):
        self.assertConfigureFails(SpectralContrast(), { 'frameSize': 0 })
        self.assertConfigureFails(SpectralContrast(), { 'frameSize': 1 })
        self.assertConfigureFails(SpectralContrast(), { 'sampleRate': 0 })
        self.assertConfigureFails(SpectralContrast(), { 'numberBands': 0 })
        self.assertConfigureFails(SpectralContrast(), { 'lowFrequencyBound': -1 })
        self.assertConfigureFails(SpectralContrast(), { 'highFrequencyBound': 40000 })
        self.assertConfigureFails(SpectralContrast(), { 'neighbourRatio': 1.5 })
        self.assertConfigureFails(SpectralContrast(), { 'staticDistribution': 1.5 })

        # lower bound cannot be larger than higher band:
        self.assertConfigureFails(SpectralContrast(), { 'lowFrequencyBound': 11000,
                                                        'highFrequencyBound': 5000 })

    def testEmpty(self):
        SC = SpectralContrast(sampleRate = 44100)
        self.assertComputeFails(SC, [])

    def testOneElement(self):
        # input spectrum must be 0.5*framesize
        SC = SpectralContrast(sampleRate = 44100)
        self.assertComputeFails(SC, [1])

    def testSpectrumSizeSmallerThanNumberOfBands(self):
        SC = SpectralContrast(sampleRate = 44100, frameSize = 4)
        sc = SC([1,1,1])
        self.assertAlmostEquals(numpy.mean(sc[0]), -2.7182817459)
        self.assertAlmostEquals(numpy.mean(sc[1]), 0)


suite = allTests(TestSpectralContrast)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
