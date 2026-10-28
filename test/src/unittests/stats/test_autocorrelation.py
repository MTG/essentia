#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

testdir = join(filedir(), 'autocorrelation')


class TestAutoCorrelation(TestCase):

    def testRegression(self):
        inputv = readVector(join(testdir, 'input_pow2.txt'))
        expected = readVector(join(testdir, 'output.txt'))

        output = AutoCorrelation()(inputv)

        self.assertAlmostEqualVector(expected, output, 1e-4)


    def testNonPowerOfTwo(self):
        inputv = readVector(join(testdir, 'octave_input.txt'))
        inputv = inputv[:234]
        expected = readVector(join(testdir, 'output_nonpow2.txt'))

        output = AutoCorrelation()(inputv)

        self.assertAlmostEqualVector(expected, output, 1e-4)


    def testOctave(self):
        inputv = readVector(join(testdir, 'octave_input.txt'))
        expected = readVector(join(testdir, 'octave_output.txt'))

        output = AutoCorrelation()(inputv)

        self.assertEqual(len(expected)/2, len(output))

        self.assertAlmostEqualVector(expected[:int(len(expected)/2)], output, 1e-4)


    def testZero(self):
        self.assertEqualVector(AutoCorrelation()(zeros(1024)), zeros(1024))

    def testEmpty(self):
        self.assertEqualVector(AutoCorrelation()([]), [])

    def testOne(self):
        self.assertAlmostEqualVector(AutoCorrelation()([0.2]), [0.04])

    def testInvalidParam(self):
        self.assertConfigureFails(AutoCorrelation(), {'normalization': 'unknown'})

    def testGeneralizedCompression2MatchesStandard(self):
        # With frequencyDomainCompression == 2, the generalized formula (|X|^k)
        # collapses to the same squared-magnitude spectrum used by the
        # non-generalized path, so both should produce the same output
        # regardless of the (zero-padded) FFT size used internally.
        # Regression for https://github.com/MTG/essentia/issues/1373.
        inputv = readVector(join(testdir, 'input_pow2.txt'))

        standard = AutoCorrelation(generalized=False)(inputv)
        generalized = AutoCorrelation(generalized=True,
                                       frequencyDomainCompression=2)(inputv)

        self.assertAlmostEqualVector(standard, generalized, 1e-4)

    def testGeneralizedInvariantToFFTSizePadding(self):
        # The generalized output must not depend on the internal (zero-padded)
        # FFT size, only on the actual signal content. Two inputs sharing the
        # same non-zero prefix but zero-padded to different total lengths
        # (forcing different internal FFT sizes) must agree on the lags within
        # that shared prefix. Regression for
        # https://github.com/MTG/essentia/issues/1373.
        #
        # For a non-integer compression exponent k, |X(w)|^k is not the
        # spectrum of a time-limited sequence, so the two FFT sizes leave a
        # small circular-aliasing tail difference: the invariance is only
        # approximate, not exact. That tail is tiny in absolute terms but can
        # dominate the near-zero values at the highest lags in *relative*
        # terms, so this uses an absolute-tolerance comparison rather than
        # assertAlmostEqualVector's per-element relative one.
        inputv = readVector(join(testdir, 'input_pow2.txt'))
        prefixLen = 32

        prefix = inputv[:prefixLen]
        # nextPowerTwo(2*40) == 128, nextPowerTwo(2*100) == 256: different
        # internal FFT sizes for the same non-zero content.
        shortSignal = prefix + [0.0] * (40 - prefixLen)
        longSignal = prefix + [0.0] * (100 - prefixLen)

        ac = AutoCorrelation(generalized=True, frequencyDomainCompression=1.5)
        shortOutput = ac(shortSignal)
        ac = AutoCorrelation(generalized=True, frequencyDomainCompression=1.5)
        longOutput = ac(longSignal)

        self.assertAlmostEqualVectorAbs(shortOutput[:prefixLen],
                                         longOutput[:prefixLen], 0.02)


suite = allTests(TestAutoCorrelation)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
