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
from numpy import mean, array, float32, square
from pathlib import Path


class TestAudio2Midi(TestCase):
    def testEmpty(self):
        self.assertComputeFails(Audio2Midi(), [])

    def testZero(self):
        pitch, loudness, message_type, midi_note, time_compensation = Audio2Midi()(
            zeros(32)
        )
        self.assertEqual(pitch, 0)
        self.assertEqual(loudness, 0.0)
        self.assertEqual(message_type, [])
        self.assertEqual(midi_note.tolist(), array([], dtype=float32).tolist())
        self.assertEqual(time_compensation.tolist(), array([], dtype=float32).tolist())

    def assessNoteList(
        self,
        reference_path: str,
        estimated: list,
        n_notes_tolerance: int = 0,
        onset_tolerance: float = 0.01,
        offset_tolerance: float = 0.01,
        midi_note_tolerance: int = 0,
    ):
        # read the expected notes file manually annotated
        expected_notes = numpy.load(join(filedir(), reference_path))
        print("Expected notes:")
        print(expected_notes)

        print("\ndiffs")
        print(array(estimated) - expected_notes[:, 1:])

        # estimate the number of notes for expected and detected
        n_detected_notes = len(estimated)
        n_expected_notes = len(expected_notes)

        # estimate the onset error for each note and estimate the mean
        local_onset_mses = array(
            [square(note[1] - estimated[int(note[0])][0]) for note in expected_notes]
        )
        global_onset_mse = mean(local_onset_mses)

        # estimate the onset error for each note and estimate the mean
        local_offset_mses = array(
            [square(note[2] - estimated[int(note[0])][1]) for note in expected_notes]
        )
        global_offset_mse = mean(local_offset_mses)

        # estimate the midi note error for each note and estimate the mean
        local_midi_note_mses = array(
            [square(note[-1] - estimated[int(note[0])][-1]) for note in expected_notes]
        )
        midi_note_mse = mean(local_midi_note_mses)

        # assert global outputs
        self.assertAlmostEqual(n_detected_notes, n_expected_notes, n_notes_tolerance)
        self.assertAlmostEqual(global_onset_mse, 0, onset_tolerance)
        self.assertAlmostEqual(global_offset_mse, 0, offset_tolerance)
        self.assertAlmostEqual(midi_note_mse, midi_note_mse, midi_note_tolerance)

        # assert local outputs
        message = "Some onset is larger than onset tolerance"
        onsets_condition = all([mse < onset_tolerance for mse in local_onset_mses])
        self.assertTrue(onsets_condition, message)

        message = "Some offset is larger than offset tolerance"
        offsets_condition = all([mse < offset_tolerance for mse in local_offset_mses])
        self.assertTrue(offsets_condition, message)

        message = "Some midi note is larger than midi note tolerance"
        midi_note_condition = all(
            [mse <= midi_note_tolerance for mse in local_midi_note_mses]
        )
        self.assertTrue(midi_note_condition, message)

    def testARealCaseWithEMajorScale(self):
        frame_size = 8192
        sample_rate = 48000
        hop_size = 64
        loudness_threshold = -40
        pitch_confidence_threshold = 0.25
        min_frequency = 103.83
        max_frequency = 659.26
        midi_buffer_duration = 0.05
        min_onset_period = 0.075
        min_offset_period = 0.2
        min_occurrence_rate = 0.5
        min_note_change_period = 0.03
        n_notes_tolerance = 0
        onset_tolerance = 0.01
        offset_tolerance = 0.05
        midi_note_tolerance = 0

        stem = "359500__mtg__sax-tenor-e-major"
        audio_path = Path("recorded") / f"{stem}.wav"
        reference_path = Path("audio2midi") / f"{stem}.npy"

        self.runARealCase(
            audio_path=audio_path,
            reference_path=reference_path,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            pitch_confidence_threshold=pitch_confidence_threshold,
            loudness_threshold=loudness_threshold,
            midi_buffer_duration=midi_buffer_duration,
            min_note_change_period=min_note_change_period,
            max_frequency=max_frequency,
            min_frequency=min_frequency,
            min_onset_period=min_onset_period,
            min_offset_period=min_offset_period,
            min_occurrence_rate=min_occurrence_rate,
            n_notes_tolerance=n_notes_tolerance,
            onset_tolerance=onset_tolerance,
            offset_tolerance=offset_tolerance,
            midi_note_tolerance=midi_note_tolerance,
        )

    def testARealCaseWithDMinorScale(self):
        frame_size = 8192
        sample_rate = 48000
        hop_size = 64
        loudness_threshold = -40
        pitch_confidence_threshold = 0.25
        min_frequency = 103.83
        max_frequency = 659.26
        midi_buffer_duration = 0.05
        min_onset_period = 0.075
        min_offset_period = 0.2
        min_occurrence_rate = 0.5
        min_note_change_period = 0.03
        n_notes_tolerance = 0
        onset_tolerance = 0.01
        offset_tolerance = 0.05
        midi_note_tolerance = 0

        stem = "359628__mtg__sax-tenor-d-minor"
        audio_path = Path("recorded") / f"{stem}.wav"
        reference_path = Path("audio2midi") / f"{stem}.npy"

        self.runARealCase(
            audio_path=audio_path,
            reference_path=reference_path,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            pitch_confidence_threshold=pitch_confidence_threshold,
            loudness_threshold=loudness_threshold,
            midi_buffer_duration=midi_buffer_duration,
            min_onset_period=min_onset_period,
            min_offset_period=min_offset_period,
            min_occurrence_rate=min_occurrence_rate,
            min_note_change_period=min_note_change_period,
            max_frequency=max_frequency,
            min_frequency=min_frequency,
            n_notes_tolerance=n_notes_tolerance,
            onset_tolerance=onset_tolerance,
            offset_tolerance=offset_tolerance,
            midi_note_tolerance=midi_note_tolerance,
        )

    def testSeparatedNotes(self):
        frame_size = 8192
        sample_rate = 44100
        hop_size = 16
        loudness_threshold = -50
        pitch_confidence_threshold = 0.8
        min_frequency = 103.83
        max_frequency = 659.26
        midi_buffer_duration = 0.05
        min_note_change_period = 0.02
        min_onset_period = 0.015
        min_offset_period = 0.015
        min_occurrence_rate = 0.05
        n_notes_tolerance = 0
        onset_tolerance = 0.01
        offset_tolerance = 0.05
        midi_note_tolerance = 0

        stem = "387517__deleted_user_7267864__saxophone-going-up"
        audio_path = Path("recorded") / f"{stem}.wav"
        reference_path = Path("audio2midi") / f"{stem}.npy"

        self.runARealCase(
            audio_path=audio_path,
            reference_path=reference_path,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            pitch_confidence_threshold=pitch_confidence_threshold,
            loudness_threshold=loudness_threshold,
            max_frequency=max_frequency,
            min_frequency=min_frequency,
            midi_buffer_duration=midi_buffer_duration,
            min_note_change_period=min_note_change_period,
            min_onset_period=min_onset_period,
            min_offset_period=min_offset_period,
            min_occurrence_rate=min_occurrence_rate,
            n_notes_tolerance=n_notes_tolerance,
            onset_tolerance=onset_tolerance,
            offset_tolerance=offset_tolerance,
            midi_note_tolerance=midi_note_tolerance,
        )

    def runARealCase(
        self,
        audio_path: str,
        reference_path: str,
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        pitch_confidence_threshold: float,
        loudness_threshold: float,
        max_frequency: float,
        min_frequency: float,
        midi_buffer_duration: float,
        min_note_change_period: float,
        min_onset_period: float,
        min_offset_period: float,
        min_occurrence_rate: float,
        n_notes_tolerance: int,
        onset_tolerance: float,
        offset_tolerance: float,
        midi_note_tolerance: int,
    ):
        filename = join(testdata.audio_dir, audio_path)
        audio = MonoLoader(filename=filename, sampleRate=sample_rate)()
        frames = FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
        step_time = hop_size / sample_rate

        # initialize audio2midi instance
        a2m = Audio2Midi(
            sampleRate=sample_rate,
            hopSize=hop_size,
            midiBufferDuration=midi_buffer_duration,
            minNoteChangePeriod=min_note_change_period,
            minOnsetCheckPeriod=min_onset_period,
            minOffsetCheckPeriod=min_offset_period,
            pitchConfidenceThreshold=pitch_confidence_threshold,
            loudnessThreshold=loudness_threshold,
            maxFrequency=max_frequency,
            minFrequency=min_frequency,
            minOccurrenceRate=min_occurrence_rate,
        )

        # define estimate bin and some counters
        nte_list = []  # note toggle event list
        n = 0
        time_stamp = 0
        n_notes = 0

        # simulates real-time process
        for frame in frames:
            # _pitch, _, _, _voiced = pitchDetect(frame)
            _pitch, _, message, midi_note, time_compensation = a2m(frame)
            time_stamp += step_time
            # print(n, time_stamp, message, midi_note, time_compensation)
            if message:
                nte_list.append(
                    [
                        n_notes,
                        time_stamp - time_compensation[1],
                        time_stamp - time_compensation[0],
                        int(midi_note[1]),
                        message,
                    ]
                )
                print(
                    f"[{n_notes}][{n}]:{(time_stamp-time_compensation[1]):.3f}, {midi2note(int(midi_note[1]))}({int(midi_note[1])})~{_pitch:.2f}Hz, {message}"  # , {time_compensation}, {midi_note}, {message}
                )
                if "note_on" in message:
                    n_notes += 1
            n += 1

        print(f"nte_list: {nte_list}")
        # from the nte_list extracts the note list using note_off messages
        note_list = self.ntes_to_notes(nte_list)
        print(f"note_list: {note_list}")

        self.assessNoteList(
            reference_path,
            note_list,
            n_notes_tolerance=n_notes_tolerance,
            onset_tolerance=onset_tolerance,
            offset_tolerance=offset_tolerance,
            midi_note_tolerance=midi_note_tolerance,
        )

    def ntes_to_notes(self, nte_list: list):
        note_list = list()
        for n, nte_message in enumerate(nte_list):
            if "note_on" in nte_message[4]:
                # extract time stamp
                start_time = nte_message[1]

                # in some cases the compensation might generate negative values
                if start_time < 0:
                    start_time = 0

                # to get the note offset it is need to get time stamps in the next message (note-off)
                if n + 1 < len(nte_list):  # when a note off message is provided
                    # define timestamp for offset
                    end_time = nte_list[n + 1][1]
                else:  # there is a non-closed note at the end
                    # define timestamp for offset
                    end_time = nte_list[-1][1]
                note = int(nte_message[3])
                # define annotation in a list
                note_list.append([float(start_time), float(end_time), note])
        return note_list


suite = allTests(TestAudio2Midi)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
