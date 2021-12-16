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
import essentia

class TestHarmonicBpm(TestCase):

    # Nothing should be computed and the resulting pool should be empty.
    def testEmpty(self):
        harmonicBpms = HarmonicBpm(bpm=100)([])
        self.assertEqualVector(harmonicBpms, [])

    # Check that illegal parameters in configuration raise asserts
    def testInvalidParam(self):
        # Test that we must give valid frequency ranges, thresholds and tolerance.
        self.assertConfigureFails(HarmonicBpm(), {'bpm': -1})
        self.assertConfigureFails(HarmonicBpm(), {'bpm': 0})
        self.assertConfigureFails(HarmonicBpm(), {'threshold': -1 })
        self.assertConfigureFails(HarmonicBpm(), {'threshold': 0 })
        self.assertConfigureFails(HarmonicBpm(), {'tolerance': -1 })

    # Simplest test case: All candidates the same returns a single value.
    def testRegressionEqualCandidateValues(self):
        testBpms = [120, 120, 120, 120, 120, 120, 120]
        harmonicBpms = HarmonicBpm(bpm=120)(testBpms)
        self.assertEqual(harmonicBpms, 120)

    # Test case with several values within a tolerance range
    def testRegressionSmallRangeBpm(self):
        testBpms = [100, 101, 102, 103, 104, 104.1]
        harmonicBpms = HarmonicBpm(bpm=100, threshold=20, tolerance=5)(testBpms)
        expectedBpm = 100
        self.assertEqual(harmonicBpms, expectedBpm)

    # Check for bpm parameter at limit of tolerance range
    # In this example: 104 + tolerance = 109
    def testRegressionBpmInsideToleranceLimit(self):
        testBpms = [100, 101, 102, 103, 104]
        harmonicBpms = HarmonicBpm(bpm=109, threshold=20, tolerance=5)(testBpms)
        expectedBpm = [104]
        self.assertEqualVector(harmonicBpms, expectedBpm)

    # Check for bpm parameter outside tolerance range
    # In this example: 104 + tolerance < 110
    def testRegressionBpmOutsideToleranceLimit(self):
        testBpms = [100, 101, 102, 103, 104]
        harmonicBpms = HarmonicBpm(bpm=110, threshold=20, tolerance=5)(testBpms)
        self.assertEqualVector(harmonicBpms, [])

    # Check the value below threshold is not included in harmonicBpms
    def testRegressionTwoOctaveRange(self):
        testBpms = [118, 120, 121, 122, 236, 240, 240, 240, 241]

        # Check first with 100 bpm and threshold+tolerance at default
        harmonicBpms = HarmonicBpm(bpm=100, threshold=20, tolerance=5)(testBpms)
        # The threshold being 20 a divisor of 120, 240, will yield these values
        # at tolerance 5.
        expectedBpm = [120, 240]
        self.assertEqualVector(harmonicBpms, expectedBpm)

        # Threshold Checks

        # Threshold is the value below which greatest common divisors of BPM are discarded
        # Check for more outputs included at lower threshold.
        harmonicBpms = HarmonicBpm(bpm=100, threshold=1, tolerance=5)(testBpms)
        expectedBpm = [118, 236]
        self.assertEqualVector(harmonicBpms, expectedBpm)

        # Check that higher threshold value lead to higher discarding rate.
        # Muliple BPMs are output and are discarded due to being below threshold.
        harmonicBpms = HarmonicBpm(bpm=100, threshold=25, tolerance=5)(testBpms)
        self.assertEqualVector(harmonicBpms, [])


        # Tolerance Checks

        # Tolerance parameter:consideration of whether two BPM values are equal or "harmonically" equal
        # Check for no outputs at reduced tolerance value
        harmonicBpms = HarmonicBpm(bpm=100, threshold=20, tolerance=4)(testBpms)
        # The threshold being 20 a divisor of 120, 240, will yield these values
        # at tolerance 4.
        expectedBpm = [120, 240]
        self.assertEqualVector(harmonicBpms,expectedBpm)

        # Increase tolerance to 10 to check if 236 is included
        harmonicBpms = HarmonicBpm(bpm=100, threshold=20, tolerance=10)(testBpms)
        # Check that multiples of threshold are included.
        expectedBpm = [120, 236]
        self.assertEqualVector(harmonicBpms, expectedBpm)

        harmonicBpms = HarmonicBpm(bpm=100, threshold=60, tolerance=5)(testBpms)
        self.assertEqualVector(harmonicBpms, [])

    # Stretch the BPM range to cover 3 octaves within audible range.
    def testRegressionThreeOctaveRange(self):
        testBpms = [100, 101, 102, 103, 104, 200, 202, 204, 206, 208, 300, 302, 304, 306, 308]
        harmonicBpms = HarmonicBpm(bpm=100)(testBpms)
        expectedBpm = [100, 200, 300]
        self.assertEqualVector(harmonicBpms, expectedBpm)

    # Check multiple BPMs with diverse octaves ranges
    def testRegressionMultipleValues(self):
        testBpms = [120, 120, 120, 120, 120, 120, 120,
                             240, 240, 240, 240, 240, 240, 240,
                             180, 180, 180, 180, 180, 180, 180,
                             90, 90, 90, 90, 90, 90, 90]

        # Check with default threshold (20) and BPM = 120.
        expectedHarmonicBpms = [90, 120, 180, 240]
        harmonicBpms = HarmonicBpm(bpm=120, threshold=20)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)

        # Check with default threshold (20) and BPM = 90.
        expectedHarmonicBpms = [90, 120, 180, 240]
        harmonicBpms = HarmonicBpm(bpm=90, threshold=20)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)

        # Run a test with a higher threshold that will lead to some outputs being discarded.

        # Check with threshold=30 and BPM = 120
        # that all multiples of 30 are included.
        expectedHarmonicBpms = [90, 120, 180, 240]
        harmonicBpms = HarmonicBpm(bpm=120, threshold=30)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)

        # Check with threshold=30 and BPM = 90
        # that all multiples of 30 are included.
        expectedHarmonicBpms = [90, 120, 180, 240]
        harmonicBpms = HarmonicBpm(bpm=90, threshold=30)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)

    # Check a range of tolerance values with wide range of BPMs.
    def testRegressionDifferentToleranceLowBpms(self):
        testBpms = [10, 20, 60, 120, 180]
        expectedHarmonicBpms = [20, 60, 120, 180]
        # We want to ensure nothing bad happens when tolerance = 0.
        harmonicBpms = HarmonicBpm(bpm=60, tolerance=0, threshold=20)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)
        # Check range tolerance values of multiples of 5.
        harmonicBpms = HarmonicBpm(bpm=60, tolerance=5, threshold=20)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)
        harmonicBpms = HarmonicBpm(bpm=60, tolerance=10, threshold=20)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)
        harmonicBpms = HarmonicBpm(bpm=60, tolerance=20, threshold=20)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)

        # Additional tests to cover lower thresholds.
        # expectedHarmonicBpms should now match the input testBpms
        expectedHarmonicBpms = [10, 20, 60, 120, 180]
        harmonicBpms = HarmonicBpm(bpm=60, tolerance=20, threshold=10)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)
        harmonicBpms = HarmonicBpm(bpm=60, tolerance=20, threshold=1)(testBpms)
        self.assertEqualVector(harmonicBpms, expectedHarmonicBpms)

    def testInvalidBpms(self):
        # Ensure that an exception is thrown if any bpm element contains a value below 1

        # BPM=1 is the lowest margin allowed.
        self.assertEqualVector(HarmonicBpm(bpm=1, threshold=1)([1]), [1])

        # Inputs with invalid BPMs.
        testBpms = [0, 1, 100]
        self.assertRaises(EssentiaException, lambda: HarmonicBpm()(testBpms))
        testBpms = [0, -1]
        self.assertRaises(EssentiaException, lambda: HarmonicBpm()(testBpms))
        testBpms = [100, 100, 100, 100, 0.9999]
        self.assertRaises(EssentiaException, lambda: HarmonicBpm()(testBpms))
        testBpms = zeros(100)
        self.assertRaises(EssentiaException, lambda: HarmonicBpm()(testBpms))

suite = allTests(TestHarmonicBpm)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

