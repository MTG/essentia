#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
        #
        # This test started to fail after the improvements
        # described in 16258bf97eedb35299450874675bf2bde6eb5aa8.
        # However, as the new way to compute the bands makes more
        # sense, lets asumme that the current behavior is correct.
        # The expected values were updated after commit
        # 3e3538080a1d4336c293f822d44cc7619b40660f

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

        expected_constrast = [-0.39171591, -0.47914168, -0.56516778,
                              -0.71739447, -0.81886953, -0.85993105]
        expected_valleys = [-6.09844828,  -6.20369911,  -8.22635174,
                            -9.10304642,  -9.83238029, -10.69903278]

        self.assertAlmostEqualVector(numpy.mean(sc, 0), expected_constrast, 1e-6)
        self.assertAlmostEqualVector(numpy.mean(valleys, 0), expected_valleys, 1e-6)



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
        # This test started to fail after the improvements
        # described in 16258bf97eedb35299450874675bf2bde6eb5aa8.
        #
        # As that change clearly improves the way to compute the
        # octave bands the it will be considered as correct.
        # This test was modified after commit
        # 3e3538080a1d4336c293f822d44cc7619b40660f

        # No contrast in any band
        spec0 = [1] * 1025

        # Contrast in the fist band only
        spec1 = [1] * 10 + [0] * 1015

        # Contrast in the fist and second bands
        spec2 = [1] * 10 + [0] * 1015
        spec2[20] = 1

        sr = 44100
        SC = SpectralContrast(sampleRate=sr)

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
        # This test creates 6 octave bands centered in
        # 57.26, 163.98, 469.57, 1344.60, 3850.23 and 11024.99 Hz.
        SC = SpectralContrast(sampleRate = 44100, frameSize = 4)

        # The following spectral bins are centered on
        # frequencies 0, 11025 and 22050 Hz.
        spec = [1,1,1]

        sc = SC(spec)

        # First check that the first 5 bands are empty.
        self.assertAlmostEquals(numpy.mean(sc[0][:5]), -2.7182817459)

        # Then check that the last bands constains one bin.
        self.assertAlmostEquals(numpy.mean(sc[0][-1]), -1)

        # In any case the valley values should be 0.
        self.assertAlmostEquals(numpy.mean(sc[1]), 0)


suite = allTests(TestSpectralContrast)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
