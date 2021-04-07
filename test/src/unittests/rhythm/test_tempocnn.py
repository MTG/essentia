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

class TestTempoCNN(TestCase):

    def regression(self, aggregation_method='majority'):
        # The expected values were self-generated with TempoCNN
        # from commit deb334ba01f849e74b0e6e8554b2e17d73bc6966
        # The retrieved tempo values were manually validated (Pablo Alonso)
        expected_global_tempo = 125.0
        expected_local_tempo = [125., 125., 125., 125.]
        expected_local_probs = [0.9679363, 0.9600502, 0.9681525, 0.9627014]

        filename = join(testdata.audio_dir, 'recorded', 'techno_loop.wav')
        model = join(testdata.models_dir, 'tempocnn', 'deeptemp_k16.pb')

        audio = MonoLoader(filename=filename, sampleRate=11025)()
        global_tempo, local_tempo, local_probs = TempoCNN(graphFilename=model,
                                                          aggregationMethod=aggregation_method)(audio)

        self.assertEqual(global_tempo, expected_global_tempo)
        self.assertEqualVector(local_tempo, expected_local_tempo)
        self.assertAlmostEqualVector(local_probs, expected_local_probs, 1e-6)

    def testMajorityVotingRegression(self):
        self.regression(aggregation_method='majority')

    def testMeanRegression(self):
        self.regression(aggregation_method='mean')

    def testMedianRegression(self):
        self.regression(aggregation_method='median')

    def testAggregationMethods(self):
        # Load an audio file without a clear rhythmic sense as we want
        # spurious local estimations for this test.
        # The goal is to assess if the global estimation is correctly
        # obtained from the local ones.
        filename = join(testdata.audio_dir, 'recorded', 'spaceambient.wav')
        model = join(testdata.models_dir, 'tempocnn', 'deeptemp_k16.pb')

        audio = MonoLoader(filename=filename, sampleRate=11025)()

        global_tempo, local_tempo, _ = TempoCNN(graphFilename=model,
                                                aggregationMethod='majority')(audio)
        self.assertEqual(global_tempo, numpy.argmax(numpy.bincount(local_tempo.astype('int'))))

        global_tempo, local_tempo, _ = TempoCNN(graphFilename=model,
                                                aggregationMethod='mean')(audio)
        self.assertEqual(global_tempo, numpy.mean(local_tempo))
        global_tempo, local_tempo, _ = TempoCNN(graphFilename=model,
                                                aggregationMethod='median')(audio)
        self.assertEqual(global_tempo, numpy.median(local_tempo))

suite = allTests(TestTempoCNN)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
