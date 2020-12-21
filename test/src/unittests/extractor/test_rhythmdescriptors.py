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

# The interface of the Rhythm descriptors algorithm is similar to the BPMHistogramDescriptors
# It has the additional outputs:
# beats_position (vector_real) - See RhythmExtractor2013 algorithm documentation
# confidence (real) - See RhythmExtractor2013 algorithm documentation
# bpm (real) - See RhythmExtractor2013 algorithm documentation
# bpm_estimates (vector_real) - See RhythmExtractor2013 algorithm documentation
# bpm_intervals (vector_real) -  See RhythmExtractor2013 algorithm documentation

from essentia_test import *

class TestRhythmDescriptors(TestCase):

    """
    def testEmpty(self):
        beats_position,confidence ,bpm ,bpm_estimates,bpm_intervals,first_peak_bpm ,first_peak_spread ,first_peak_weight,second_peak_bpm ,second_peak_spread,second_peak_weight, histogram  = RhythmDescriptors()([])
        self.assertEqualVector(beats_position, [0.] * len(beats_position))
        self.assertEqual(confidence,0.0)
        self.assertEqual(bpm,0.0)
        self.assertEqualVector(bpm_estimates, [0.] * len(bpm_estimates))
        self.assertEqualVector(bpm_intervals, [0.] * len(bpm_intervals))
        self.assertEqual(first_peak_bpm,0.0)
        self.assertEqual(first_peak_spread,0.0)
        self.assertEqual(first_peak_weight,0.0)
        self.assertEqual(second_peak_bpm,0.0)
        self.assertEqual(second_peak_spread,0.0)
        self.assertEqual(second_peak_weight,0.0)
        self.assertEqualVector(histogram, [0.] * len(histogram))
    """


    def testZero(self):
        beats_position,confidence ,bpm ,bpm_estimates,bpm_intervals,first_peak_bpm ,first_peak_spread ,first_peak_weight,second_peak_bpm ,second_peak_spread,second_peak_weight, histogram  = RhythmDescriptors()([0])
        self.assertEqualVector(beats_position, [0.] * len(beats_position))
        self.assertEqual(confidence,0.0)
        self.assertEqual(bpm,0.0)
        self.assertEqualVector(bpm_estimates, [0.] * len(bpm_estimates))
        self.assertEqualVector(bpm_intervals, [0.] * len(bpm_intervals))
        self.assertEqual(first_peak_bpm,0.0)
        self.assertEqual(first_peak_spread,0.0)
        self.assertEqual(first_peak_weight,0.0)
        self.assertEqual(second_peak_bpm,0.0)
        self.assertEqual(second_peak_spread,0.0)
        self.assertEqual(second_peak_weight,0.0)
        self.assertEqualVector(histogram, [0.] * len(histogram))

    """
    def testOne(self):
        beats_position,confidence ,bpm ,bpm_estimates,bpm_intervals,first_peak_bpm ,first_peak_spread ,first_peak_weight,second_peak_bpm ,second_peak_spread,second_peak_weight, histogram  = RhythmDescriptors()([0.5])
        self.assertEqualVector(beats_position, [0.] * len(beats_position))
        self.assertEqual(confidence,0.0)
        self.assertEqual(bpm,0.0)
        self.assertEqualVector(bpm_estimates, [0.] * len(bpm_estimates))
        self.assertEqualVector(bpm_intervals, [0.] * len(bpm_intervals))
        self.assertEqual(first_peak_bpm,0.0)
        self.assertEqual(first_peak_spread,0.0)
        self.assertEqual(first_peak_weight,0.0)
        self.assertEqual(second_peak_bpm,0.0)
        self.assertEqual(second_peak_spread,0.0)
        self.assertEqual(second_peak_weight,0.0)
        self.assertEqualVector(histogram, [0.] * 120 + [1.] + [0.] * (len(histogram)-121))


    def testAbnormalValues(self):
        bpms = [-100, 300]
        intervals = []
        for bpm in bpms:
           intervals.append(60. / bpm)

           beats_position,confidence ,bpm ,bpm_estimates,bpm_intervals,first_peak_bpm ,first_peak_spread ,first_peak_weight,second_peak_bpm ,second_peak_spread,second_peak_weight, histogram  = RhythmDescriptors()([intervals])
           self.assertEqualVector(beats_position, [0.] * len(beats_position))
           self.assertEqual(confidence,0.0)
           self.assertEqual(bpm,0.0)
           self.assertEqualVector(bpm_estimates, [0.] * len(bpm_estimates))
           self.assertEqualVector(bpm_intervals, [0.] * len(bpm_intervals))
           self.assertEqual(first_peak_bpm,0.0)
           self.assertEqual(first_peak_spread,0.0)
           self.assertEqual(first_peak_weight,0.0)
           self.assertEqual(second_peak_bpm,0.0)
           self.assertEqual(second_peak_spread,0.0)
           self.assertEqual(second_peak_weight,0.0)
           self.assertEqualVector(histogram, [0.] * 120 + [1.] + [0.] * (len(histogram)-121))

    def testRounding(self):
        #bpm1, weight1, spread1, bpm2, weight2, spread2, histogram = RhythmDescriptors()([60. / 100.5])
        beats_position,confidence ,bpm ,bpm_estimates,bpm_intervals,first_peak_bpm ,first_peak_spread ,first_peak_weight,second_peak_bpm ,second_peak_spread,second_peak_weight, histogram  =  RhythmDescriptors()([60. / 100.5])
        self.assertEqual(bpm1, 101)
 

    def testRegression(self):
        bpms = [118, 119, 120, 120, 121, 122, 98, 99, 99, 100, 100, 100, 100, 101, 101, 102]
        intervals = []
        for bpm in bpms:
           intervals.append(60. / bpm)
           intervals.append(0) # Add an extra zero to check if it gets properly dropped
           #bpm1, weight1, spread1, bpm2, weight2, spread2, histogram = RhythmDescriptors()(intervals)
           beats_position,confidence ,bpm ,bpm_estimates,bpm_intervals,first_peak_bpm ,first_peak_spread ,first_peak_weight,second_peak_bpm ,second_peak_spread,second_peak_weight, histogram  =   RhythmDescriptors()(intervals)
           self.assertAlmostEqual(first_peak_bpm, 100, 1e-5)
           self.assertAlmostEqual(first_peak_weight, 0.5, 1e-5)
           self.assertAlmostEqual(first_peak_spread, 0.2, 1e-5)
           self.assertAlmostEqual(second_peak_bpm, 120, 1e-5)
           self.assertAlmostEqual(second_peak_weight, 0.25, 1e-5)
           self.assertAlmostEqual(second_peak_spread, 0.333333, 1e-5)

        correct_histogram = [0.] * len(histogram)
        for bpm in bpms:
           correct_histogram[bpm] += 1./len(bpms)
        self.assertEqualVector(histogram, histogram)
    """

suite = allTests(TestRhythmDescriptors)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)


