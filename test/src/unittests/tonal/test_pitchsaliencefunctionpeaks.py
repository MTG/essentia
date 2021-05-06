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


from numpy import *
from essentia_test import *

class TestPitchSalienceFunctionPeaks(TestCase):
  
    def testInvalidParam(self):
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'binResolution': -1})
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'maxFrequency': -1})
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'minFrequency': -1})
        self.assertConfigureFails(PitchSalienceFunctionPeaks(), {'referenceFrequency': -1})          

    def testEmpty(self):
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunctionPeaks()([]))
    
    # Provide a single input peak with a unit magnitude at the reference frequency, and 
    # validate that the output salience function has only one non-zero element at the first bin.
    def testSinglePeak(self):
        freq_speaks = [55]
        mag_speaks = [1]
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
        self.assertEqualVector(bins, [0.])
        self.assertEqualVector(values, [1.])

    # Provide 3 input peaks with a unit magnitude and 
    # validate that the output salience with previously calcualted values
    def testRegression3Peaks(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        # For a single frequency 55 Hz with unitary  amplitude the 10 non zero salience function values are the following
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
        self.assertEqualVector(bins, [ 1., 103., 315., 195., 125.,  75.,  37.] )
        self.assertAlmostEqualVectorFixedPrecision(values, [1.1899976, 1., 1., 0.8, 0.64000005, 0.512, 0.40960002], 6)

    # Checks on config. values binResolution being too high for the input in question
    def testRegression3PeaksHighBinResolution(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        calculatedPitchSalience = PitchSalienceFunction(binResolution=2400)(freq_speaks,mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
        self.assertEqualVector(bins, [] )
        self.assertEqualVector(values, [])

    # Checks on config. values maxFrequency being too low for a given set of spectral peaks
    def testRegressionBadMaxFreq(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks,mag_speaks)
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunctionPeaks(maxFrequency=50)(calculatedPitchSalience))

    # Checks on config. values minFrequency being too high for a given set of spectral peaks
    def testRegressionBadMinFreq(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunctionPeaks(minFrequency=20000)(calculatedPitchSalience))

    # Checks on config. values referenceFrequency being too high for a given set of spectral peaks
    def testRegressionBadRefFreq(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        calculatedPitchSalience = PitchSalienceFunction()(freq_speaks, mag_speaks)
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunctionPeaks(referenceFrequency=20000)(calculatedPitchSalience))
        
    # As in testRegression3Peaks but with Harmonic Weight = 0, non multiple freq. inputs
    def testRegression3PeaksHw0UnharmonicInput(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        expectedBins =  [0., 103., 315.]
        expectedValues = [1., 1., 1.]
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=0)(freq_speaks, mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
        self.assertAlmostEqualVectorFixedPrecision(bins, expectedBins, 6)
        self.assertAlmostEqualVectorFixedPrecision(values, expectedValues, 6)
    
    # As in testRegression3Peaks but with Harmonic Weight = 0, multiple freq. inputs
    def testRegression3PeaksHw0HarmonicInput(self):
        freq_speaks = [110, 220]
        mag_speaks = [1, 1 ]
        expectedBins =  [120., 240.]
        expectedValues = [1., 1.]
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=0)(freq_speaks, mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
        self.assertAlmostEqualVectorFixedPrecision(bins, expectedBins, 6)
        self.assertAlmostEqualVectorFixedPrecision(values, expectedValues, 6)

    # As in testRegression3Peaks but with Harmonic Weight = 0,  non multiple freq. inputs
    def testRegression3PeaksHw1UnHarmonicInput(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=1)(freq_speaks, mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)
        self.assertEqualVector(bins, [2., 37., 75., 103., 125., 195., 315.])
        self.assertAlmostEqualVectorFixedPrecision(values, [1.6984011, 1., 1., 1., 1., 1., 1.], 6)
    
    # As in testRegression3Peaks but with Harmonic Weight = 1, multiple freq. inputs
    def testRegression3PeaksHw1HarmonicInput(self):
        freq_speaks = [110, 220]
        mag_speaks = [1, 1 ]
        expectedBins =  [0., 120.,  50., 240.]
        expectedValues = [2., 2., 1., 1.]
        calculatedPitchSalience = PitchSalienceFunction(harmonicWeight=1)(freq_speaks, mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience)

    # Legal Checks on config. value combination
    def testMinMaxFreqError(self):
        freq_speaks = [55, 100, 340]
        mag_speaks = [1, 1, 1]
        self.assertRaises(RuntimeError, lambda: PitchSalienceFunctionPeaks(minFrequency=150,maxFrequency=150)([]))

    def testRegressionDifferentPeaks(self):
        freq_speaks = [55, 85]
        mag_speaks = [1, 1]
        calculatedPitchSalience1 = PitchSalienceFunction()(freq_speaks,mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience1)  
        self.assertEqualVector(bins,  [0.,  75.] )
        self.assertEqualVector(values, [1., 1.])

        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience1)
        mag_speaks = [0.5, 2]
        calculatedPitchSalience2 = PitchSalienceFunction()(freq_speaks,mag_speaks)
        bins, values = PitchSalienceFunctionPeaks()(calculatedPitchSalience2)
        self.assertEqualVector(bins,  [75., 0.] )
        self.assertEqualVector(values, [2., 0.5])


suite = allTests(TestPitchSalienceFunctionPeaks)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
