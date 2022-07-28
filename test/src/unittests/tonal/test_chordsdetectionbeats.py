# Copyright (C) 2006-2022  Music Technology Group - Universitat Pompeu Fabra
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
import numpy as np


class TestChordsDetectionBeats(TestCase):
    def testEmpty(self):
        with self.assertRaises(EssentiaException):
            ChordsDetectionBeats()(np.zeros(shape=(0, 12), dtype=np.single), np.zeros(shape=(0,), dtype=np.single))

        with self.assertRaises(EssentiaException):
            ChordsDetectionBeats()(
                np.random.rand(100, 12).astype(dtype=np.single), np.zeros(shape=(0,), dtype=np.single))

        chords, strength = ChordsDetectionBeats()(
            np.zeros(shape=(0, 12), dtype=np.single), [0.75 * i for i in range(10)]
        )
        self.assertEqual(len(chords), 0)

    @staticmethod
    def get_hpcp(audio_path):
        audio = MonoLoader(filename=audio_path)()
        w = Windowing(type='blackmanharris62', size=4096)
        spec = Spectrum(size=4096)
        s_peak = SpectralPeaks(minFrequency=30)
        hpcp_algo = HPCP()
        pcp = []
        for frame in FrameGenerator(audio, frameSize=4096, hopSize=2048):
            peak_freq, peak_mag = s_peak(spec(w(frame)))
            pcp.append(hpcp_algo(peak_freq, peak_mag))

        return np.array(pcp, dtype=np.single)


    def testSynthesized(self):
        pcp = TestChordsDetectionBeats.get_hpcp(
            join(testdata.audio_dir, "generated", "synthesised", "chord_detection", "Chord_Detection_Test.flac"))

        ticks = [i * 0.75 for i in range(25)]

        chords, strength = ChordsDetectionBeats()(pcp, ticks)
        chords_groundtruth = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        chords_groundtruth = chords_groundtruth + [s+"m" for s in chords_groundtruth]

        for detected_chord, expected_chord, strength in zip(chords, chords_groundtruth, strength):
            self.assertEqual(detected_chord, expected_chord)
            self.assertTrue(strength > 0.8)


    def testRecorded(self):
        pcp = TestChordsDetectionBeats.get_hpcp(
            join(testdata.audio_dir, "recorded", "guitar_triads.flac")
        )
        # FIXME The algorithm fails in detecting major/minor for one of the
        # chords in the sequence, even though the root note is identified
        # correctly.

        ticks = [2.2, 5.7, 9.0, 12.0, 16.0, 18.8, 22.6, 25.4, 28.7, 31.6,
                 34.7, 38.5, 40.6, 43.0, 45.2, 48.5, 50.5, 53.3, 56.5]

        chords, strength = ChordsDetectionBeats()(pcp, ticks)
        strength_threshold = [0.8] * 18
        chords_groundtruth = ['Em', 'Am', 'Dm', 'G', 'C', 'F#', 'Bm', 'B',
                              'Em', 'Em', 'E', 'A', 'D', 'E', 'E', 'A', 'D', 'Em']
        strength_threshold[3] = 0.4

        for detected_chord, expected_chord, s, threshold in \
                zip(chords, chords_groundtruth, strength, strength_threshold):
            self.assertEqual(detected_chord, expected_chord)
            self.assertTrue(s > threshold)


suite = allTests(TestChordsDetectionBeats)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
