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
from numpy import sin, pi, mean, random, array, float32


class TestPitch2Midi(TestCase):
    def testEmpty(self):
        self.assertComputeFails(Pitch2Midi(), -1, 0)

    def testZero(self):
        message_type, midi_note, time_compensation = Pitch2Midi()(0, 0)
        self.assertEqual(message_type, [])
        self.assertEqual(midi_note.tolist(), array([], dtype=float32).tolist())
        self.assertEqual(time_compensation.tolist(), array([], dtype=float32).tolist())

    def testOnset(self):
        sample_rate = 44100
        hop_size = 128
        onset_compensation = 0.075
        pitch = 440.0
        nblocks_for_onset = round(onset_compensation / (hop_size / sample_rate))
        pitches = [pitch] * nblocks_for_onset
        voicings = [1] * nblocks_for_onset
        expected_message_type = ["note_on"]

        self.runTest(sample_rate, hop_size, pitches, voicings, expected_message_type)

    def testUnvoicedFrame(self):
        sample_rate = 44100
        hop_size = 128
        onset_compensation = 0.075
        minNoteChangePeriod = 0.03
        nblocks_for_onset = round(onset_compensation / (hop_size / sample_rate))
        nblocks_for_offset = round(minNoteChangePeriod / (hop_size / sample_rate)) + 1
        pitches = ([440.0] * nblocks_for_onset) + ([0] * nblocks_for_offset)
        voicings = ([1] * nblocks_for_onset) + ([0] * nblocks_for_offset)
        expected_message_type = ["note_off"]

        self.runTest(sample_rate, hop_size, pitches, voicings, expected_message_type)

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
        expected_message_type = ["note_off", "note_on"]

        self.runTest(sample_rate, hop_size, pitches, voicings, expected_message_type)

    def testContinuousChromaticSequence(self):
        sample_rate = 44100
        hop_size = 128
        onset_compensation = 0.075
        minNoteChangePeriod = 0.03
        midi_buffer_duration = 0.015
        min_occurrence_rate = 0.5
        min_occurrence_period = midi_buffer_duration * min_occurrence_rate
        nblocks_for_onset = round(onset_compensation / (hop_size / sample_rate))
        nblocks_for_offset = round(minNoteChangePeriod / (hop_size / sample_rate))
        nblocks_for_transition = round(min_occurrence_period / (hop_size / sample_rate))
        n_notes = 12
        midi_notes = list(range(69, 69 + n_notes))
        #print(midi_notes)
        pitches = [midi2hz(note) for note in midi_notes]
        pitch_list = list()
        for pitch in pitches:
            pitch_list += [pitch] * (nblocks_for_transition + nblocks_for_onset)
        pitch_list += [pitch] * (nblocks_for_offset + 1)
        voicings = [1] * n_notes * (nblocks_for_onset + nblocks_for_transition)
        voicings += [0] * (nblocks_for_offset + 2)
        #print(len(pitch_list), len(voicings))
        expected_message_type = ["note_off"]
        self.runTest(sample_rate, hop_size, pitch_list, voicings, expected_message_type)

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
            time_compensations,
            message_types,
        ) = ([] for i in range(3))

        for n, (pitch, voiced) in enumerate(zip(pitches, voicings)):
            message, midi_note, time_compensation = p2m(pitch, voiced)
            #print(n, message, midi_note, time_compensation)
            message_types.append(message)
            midi_notes += [midi_note]
            time_compensations += [time_compensation]
        self.assertEqual(message_types[-1], expected_value)


suite = allTests(TestPitch2Midi)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
