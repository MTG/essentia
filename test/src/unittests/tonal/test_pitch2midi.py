#!/usr/bin/env python

# Copyright (C) 2006-2024  Music Technology Group - Universitat Pompeu Fabra
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
from numpy import sin, pi, mean, random


class TestPitch2Midi(TestCase):
    # def testEmpty(self):
    #     self.assertComputeFails(Pitch2Midi(), None, None)

    # def testZero(self):
    #     midi_note, dmidi_note, onset_compensation, offset_compensation, message_type = (
    #         Pitch2Midi()(0, 0)
    #     )
    #     self.assertEqual(midi_note, 0)
    #     self.assertEqual(dmidi_note, 0)
    #     self.assertEqual(onset_compensation, 0)
    #     self.assertEqual(offset_compensation, 0)
    #     self.assertEqual(message_type, -1)

    def testOnset(self):
        sample_rate = 44100
        hop_size = 128
        onset_compensation = 0.075
        nblocks_for_onset = round(onset_compensation / (hop_size / sample_rate))
        print(f"nblocks_for_onset: {nblocks_for_onset}")
        pitches = [440.0] * nblocks_for_onset
        voicings = [1] * nblocks_for_onset

        self.runTest(sample_rate, hop_size, pitches, voicings, 1)

    def testUnvoicedFrame(self):
        sample_rate = 44100
        hop_size = 128
        onset_compensation = 0.075
        minNoteChangePeriod = 0.03
        nblocks_for_onset = round(onset_compensation / (hop_size / sample_rate))
        nblocks_for_offset = round(minNoteChangePeriod / (hop_size / sample_rate)) + 1
        pitches = ([440.0] * nblocks_for_onset) + ([0] * nblocks_for_offset)
        voicings = ([1] * nblocks_for_onset) + ([0] * nblocks_for_offset)

        self.runTest(sample_rate, hop_size, pitches, voicings, 0)

    def testOffset(self):
        sample_rate = 44100
        hop_size = 128
        onset_compensation = 0.075
        min_occurrence_rate = 0.015 / 2
        nblocks_for_onset = round(onset_compensation / (hop_size / sample_rate))
        nblocks_for_offset = round(min_occurrence_rate / (hop_size / sample_rate))
        midi_notes = [69, 70]
        pitches = [midi2hz(note) for note in midi_notes]
        pitches = ([pitches[0]] * nblocks_for_onset) + (
            [pitches[1]] * nblocks_for_offset
        )
        voicings = [1] * (nblocks_for_onset + nblocks_for_offset)

        self.runTest(sample_rate, hop_size, pitches, voicings, 2)

    def runTest(
        self,
        sample_rate: int,
        hop_size: int,
        pitches: list,
        voicings: list,
        expected_value: int,
    ):
        p2m = Pitch2Midi(sampleRate=sample_rate, hopSize=hop_size)
        (
            midi_notes,
            dmidi_notes,
            onset_compensation,
            offset_compensation,
            message_type,
        ) = ([] for i in range(5))

        for n, (pitch, voiced) in enumerate(zip(pitches, voicings)):
            # print(f"pitch: {pitch}")
            # print(f"voiced: {voiced}")
            note, dnote, on_comp, off_comp, message = p2m(pitch, voiced)
            # print(n, note, dnote, on_comp, off_comp, message)
            midi_notes += [note]
            dmidi_notes += [dnote]
            onset_compensation += [on_comp]
            offset_compensation += [off_comp]
            message_type += [message]
        self.assertEqual(message_type[-1], expected_value)

    # def testSine(self):
    #     sr = 44100
    #     size = sr * 1
    #     freq = 440
    #     signal = [sin(2.0 * pi * freq * i / sr) for i in range(size)]
    #     self.runTest(signal, sr, freq)

    # def runTest(self, signal, sr, freq, pitch_precision=1, conf_precision=0.1):
    #     frameSize = 1024
    #     hopsize = frameSize

    #     frames = FrameGenerator(signal, frameSize=frameSize, hopSize=hopsize)
    #     win = Windowing(type="hann")
    #     pitchDetect = PitchYinFFT(frameSize=frameSize, sampleRate=sr)
    #     pitch = []
    #     confidence = []
    #     for frame in frames:
    #         spec = Spectrum()(win(frame))
    #         f, conf = pitchDetect(spec)
    #         # TODO process pitch with Pitch2Midi instance
    #         pitch += [f]
    #         confidence += [conf]
    #     self.assertAlmostEqual(mean(f), freq, pitch_precision)
    #     self.assertAlmostEqual(mean(confidence), 1, conf_precision)


suite = allTests(TestPitch2Midi)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
