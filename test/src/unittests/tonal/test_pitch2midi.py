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
            # print(n, message, midi_note, time_compensation)
            # print(n, note, dnote, on_comp, off_comp, message)
            message_types.append(message)
            midi_notes += [midi_note]
            time_compensations += [time_compensation]
        self.assertEqual(message_types[-1], expected_value)

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
