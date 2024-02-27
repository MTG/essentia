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
import os.path

from essentia_test import *
import numpy as np


class TestTonalExtractor(TestCase):

    def testEmpty(self):
        # Test if the algorithm handles an empty input signal correctly
        with self.assertRaises(RuntimeError):
            chords_changes_rate, _, _, chords_number_rate, _, _, _, _, _, _, _, key_strength = TonalExtractor()(np.array([], dtype=np.float32))

    def testSilence(self):
        # In this test we jiuts check three of the output parameters of type real
        silence_vec = np.zeros(44100, dtype=np.single)
        chords_changes_rate, _, _, chords_number_rate, _, _, _, _, _, _, _, key_strength = TonalExtractor()(silence_vec)
        self.assertEqual(chords_changes_rate, 0.0)
        self.assertGreaterEqual(chords_number_rate, 0.0)
        self.assertEqual(key_strength, 0.0)

    def testInvalidParameters(self):
        # Test if the algorithm handles invalid parameters correctly
        extractor = TonalExtractor()

        # Test case 1: Negative frameSize
        with self.subTest(msg="Negative frameSize"):
            with self.assertRaises(RuntimeError):
                extractor.configure(frameSize=-1, hopSize=2048, tuningFrequency=440.0)

        # Test case 2: Negative hopSize
        with self.subTest(msg="Negative hopSize"):
            with self.assertRaises(RuntimeError):
                extractor.configure(frameSize=4096, hopSize=-1, tuningFrequency=440.0)

        # Test case 3: Negative tuningFrequency
        with self.subTest(msg="Negative tuningFrequency"):
            with self.assertRaises(RuntimeError):
                extractor.configure(frameSize=4096, hopSize=2048, tuningFrequency=-440.0)

        # Test case 4: Zero frameSize and hopSize
        with self.subTest(msg="Zero frameSize and hopSize"):
            with self.assertRaises(RuntimeError):
                extractor.configure(frameSize=0, hopSize=0, tuningFrequency=440.0)

        # Test case 5: Zero frameSize
        with self.subTest(msg="Zero frameSize"):
            with self.assertRaises(RuntimeError):
                extractor.configure(frameSize=0, hopSize=2048, tuningFrequency=440.0)

        # Test case 6: Zero hopSize
        with self.subTest(msg="Zero hopSize"):
            with self.assertRaises(RuntimeError):
                extractor.configure(frameSize=4096, hopSize=0, tuningFrequency=440.0)

        # Test case 7: Non-negative parameters
        with self.subTest(msg="Valid parameters"):
            # This should not raise an exception
            extractor.configure(frameSize=4096, hopSize=2048, tuningFrequency=440.0)

    def testRandomInput(self):
        n = 10
        for _ in range(n):
            rand_input = np.random.random(88200).astype(np.single) * 2 - 1
            result = TonalExtractor()(rand_input)
            expected_result = np.sum(rand_input * rand_input) ** 0.67
            self.assertAlmostEqual(result[0], expected_result, 9.999e+02)

    def testRegression(self):
        frameSizes = [256, 512, 1024, 2048, 4096, 8192] 
        hopSizes = [128, 256, 512, 1024, 2048, 4096]

        input_filename = join(testdata.audio_dir, "recorded", "dubstep.wav")  # Replace 'testdata' with actual path
        realAudio = MonoLoader(filename=input_filename)()

        # Iterate through pairs of frameSize and corresponding hopSize
        # TODO: Extend loop to try different tuningFrequency values
        for fs, hs in zip(frameSizes, hopSizes):
            with self.subTest(frameSize=fs, hopSize=hs):
                # Process the algorithm on real audio with the current frameSize and hopSize
                te = TonalExtractor()
                te.configure(frameSize=fs, hopSize=hs)
                chords_changes_rate, _, _, chords_number_rate, _, _, _, _, _, _, _, key_strength= te(realAudio)

                # Perform assertions on one or more outputs
                # Example: Assert that chords_changes_rate is a non-negative scalar
                self.assertIsInstance(chords_changes_rate, (int, float))
                self.assertGreaterEqual(chords_changes_rate, 0)
                self.assertIsInstance(chords_number_rate, (int, float))
                self.assertGreaterEqual(chords_number_rate, 0)
                self.assertIsInstance(key_strength, (int, float))
                self.assertGreaterEqual(key_strength, 0)
                # You can add more assertions on other outputs as needed

    def testRealAudio(self):

        # These reference values could also be compared with th results of tonal extractors of alternative a
        # audio libraries (e.g. MadMom, libs fromn Alexander Lerch etc.)
        # ccr = chord changes rate ; cnr = chord number rate; ks = key strength
        mozart_ccr = 0.03400309011340141
        mozart_cnr = 0.010819165036082268
        mozart_ks = 0.8412253260612488
        
        vivaldi_ccr = 0.052405908703804016
        vivaldi_cnr = 0.004764173645526171
        vivaldi_ks = 0.7122617959976196

        thresh = 0.5

        def test_on_real_audio(path, ccr, cnr, ks):
            realAudio = MonoLoader(filename=path)()

            # Use default configuration of algorothm
            # This function could be extended to test for more outputs
            # TODO: Extend to test non-scalar and string outputs: 
            # i.e. chords_histogram, chords_progression, chords_scale, chords_strength
            # hpcp, hpcp_highres, key_key and key_scale
            te = TonalExtractor()
            chords_changes_rate, _, _, chords_number_rate, _, _, _, _, _, _, _, key_strength= te(realAudio)
            self.assertIsInstance(chords_changes_rate, (int, float))
            self.assertGreaterEqual(chords_changes_rate, 0)
            self.assertAlmostEqual(chords_changes_rate, ccr, thresh)
            self.assertIsInstance(chords_number_rate, (int, float))
            self.assertGreaterEqual(chords_number_rate, 0)
            self.assertAlmostEqual(chords_number_rate, cnr, thresh)
            self.assertIsInstance(key_strength, (int, float))
            self.assertGreaterEqual(key_strength, 0)
            self.assertAlmostEqual(key_strength, ks, thresh)

        test_on_real_audio(join(testdata.audio_dir, "recorded", "mozart_c_major_30sec.wav"), mozart_ccr, mozart_cnr, mozart_ks)
        test_on_real_audio(join(testdata.audio_dir, "recorded", "Vivaldi_Sonata_5_II_Allegro.wav"), vivaldi_ccr, vivaldi_cnr, vivaldi_ks)
  
suite = allTests(TestTonalExtractor)

if __name__ == '__main__':
    unittest.main()
