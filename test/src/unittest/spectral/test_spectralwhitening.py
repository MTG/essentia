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

class TestSpectralWhitening(TestCase):
    # Note: Some of the following tests do not use a spectral peaks input that
    # is derived from the spectrum input. While this is indeed not real input,
    # it has no detrimental effects to the running of the algorithm (i.e. makes
    # it crash). The input spectrum only has an effect on how the algorithm
    # performs whitening. Valid input should be provided if testing the
    # accuracy of the algorithm (i.e. performance tests). -rtoscano

    def testWhiteness(self):
        inputSize = 512
        sampleRate = 44100
        nPeaks = 3

        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin5000.wav'),
                           sampleRate = sampleRate)()

        fc = FrameCutter(frameSize = inputSize,
                         hopSize = inputSize)

        windowing = Windowing(type = 'blackmanharris62')
        spec = Spectrum(size=inputSize)


        speaks = SpectralPeaks(sampleRate = 44100,
                               maxPeaks = nPeaks,
                               maxFrequency = sampleRate/2,
                               minFrequency = 0,
                               magnitudeThreshold = 0,
                               orderBy = 'magnitude')
        specWhitener = SpectralWhitening(maxFrequency=sampleRate/2, sampleRate=sampleRate)

        frame = fc(audio)
        whitenedMagSums = [0]*nPeaks
        counter = 0
        while len(frame) != 0:
            spectrum = spec(windowing(frame))
            (freqs, mags) = speaks(spectrum)
            whitenedMags = specWhitener(spectrum, freqs, mags)

            for i in xrange(0, len(whitenedMags)):
                whitenedMagSums[i] += whitenedMags[i]
            counter += 1

            frame = fc(audio)

        whitenedMagAvgs = [x/counter for x in whitenedMagSums]

        expected = [0.05626755335908027, 0.0030654285443056555, 0.0041451468459836284]
        self.assertAlmostEqualVector(whitenedMagAvgs, expected, 1e-5)

    def testEmpty(self):
        expected = []
        self.assertEqualVector(SpectralWhitening()([], [], []), expected)

    def testZero(self):
        expected = [1.]*3
        # Arbitrary sizes
        actual = SpectralWhitening()([0]*100, [0]*3, [0]*3)
        self.assertEqualVector(actual, expected)

    def testInvalidInput(self):
        # input freqs and mags have different size
        self.assertComputeFails(SpectralWhitening(), [0]*3, [0]*99, [0]*100)

    def testOne(self):
        expected = [0.9828788638]
        actual = SpectralWhitening()([10]*1, [30]*1, [10]*1)

        # Checks for a single arbitrary value
        self.assertAlmostEqualVector(actual, expected)

    def testMaxFrequency(self):
        # Only include frequencies that are above maxFrequency, which results
        # in an unchanged magnitude vector
        freqs = [101, 102]
        mags = [1, 1]
        whitener = SpectralWhitening(maxFrequency=100)
        actual = whitener([1]*100, freqs, [1, 1])
        self.assertEqualVector(actual, mags)

    def testBoundaryCase(self):
        # one peak is under maxFrequency bound and the other is outside
        maxFrequency = 100

        # The following bound looks weird because it is calculated in the same
        # way as it is in the algorithm. I don't really understand the 1.2, but
        # the 100 is to make sure that the frequency falls within the domain
        # of points fed to the internal BPF algorithm since 100 is used as the
        # resolution of the BPF. -rtoscano
        bound = maxFrequency * 1.2 - 100
        bound -= 1e-5 # imprecision due to rounding errors between platforms

        freqs = [bound, bound+1]
        mags = [1, 1]
        whitener = SpectralWhitening(maxFrequency=maxFrequency)
        expected = [0.9885531068, 1.] # Only the first mag is whitened
        actual = whitener(range(0,100), freqs, mags)
        self.assertAlmostEqualVector(actual, expected)


suite = allTests(TestSpectralWhitening)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
