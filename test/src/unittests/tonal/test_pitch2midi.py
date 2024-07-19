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
from numpy import sin, pi, mean, random, array, float32, square


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
        # print(midi_notes)
        pitches = [midi2hz(note) for note in midi_notes]
        pitch_list = list()
        for pitch in pitches:
            pitch_list += [pitch] * (nblocks_for_transition + nblocks_for_onset)
        pitch_list += [pitch] * (nblocks_for_offset + 1)
        voicings = [1] * n_notes * (nblocks_for_onset + nblocks_for_transition)
        voicings += [0] * (nblocks_for_offset + 2)
        # print(len(pitch_list), len(voicings))
        expected_message_type = ["note_off"]
        self.runTest(sample_rate, hop_size, pitch_list, voicings, expected_message_type)

    def runTest(
        self,
        sample_rate: int,
        hop_size: int,
        pitches: list,
        voicings: list,
        expected_value: int,
        expected_idx: int = -1,
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
            message_types.append(message)
            midi_notes += [midi_note]
            time_compensations += [time_compensation]
        self.assertEqual(message_types[expected_idx], expected_value)
        # TODO: think in an strategy similar to real cases that assess the number of notes, messages and timestamp tolerance

    def testARealCase(self):

        frameSize = 8192
        sample_rate = 44100
        hopSize = 128
        loudness_threshold = -50
        pitch_confidence_threshold = 0.25

        filename = join(testdata.audio_dir, "recorded", "sax-test_min.wav")
        if sys.platform == "darwin":
            import soundfile as sf

            audio, _ = sf.read(filename, dtype="float32")
            audio = audio[:, 0]
        else:
            audio = MonoLoader(filename=filename, sampleRate=44100)()
        frames = FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)

        n_frames = (audio.shape[0] - (frameSize - hopSize)) / hopSize

        # print(f"n_frames: {n_frames}")
        # pitchDetect = Audio2Pitch(
        #     frameSize=frameSize,
        #     sampleRate=sample_rate,
        #     pitchConfidenceThreshold=0.15,
        #     loudnessThreshold=loudness_threshold,
        # )

        win = Windowing(normalized=False, zeroPhase=False)
        spec = Spectrum()
        pitchDetect = PitchYinFFT(
            frameSize=frameSize,
            sampleRate=sample_rate,
            weighting="custom",
            tolerance=1.0,
        )
        rmsDetect = RMS()

        n_outputs = len(pitchDetect.outputNames())
        pitch, confidence = ([] for _ in range(n_outputs))
        loudness = []
        voiced = []
        n = 0
        for frame in frames:
            # print(n, end="\r")
            # print(frame.shape)
            # out = pitchDetect(frame)
            _spec = spec(win(frame))
            f, conf = pitchDetect(_spec)
            pitch += [f]
            confidence += [conf]
            loudness += [rmsDetect(frame)]
            if (
                amp2db(loudness[n]) > loudness_threshold
                and confidence[n] > pitch_confidence_threshold
            ):
                voiced += [1]
            else:
                voiced += [0]
            n += 1

        expected_value = ["note_off"]
        expected_idx = 2869
        # TODO: create an annotation file for assessing the note toggle events
        # TODO: apply audio2pitch module and assess with note metrics
        self.runTest(sample_rate, hopSize, pitch, voiced, expected_value, expected_idx)

    def testARealCaseWithEMajorScale(self):
        frame_size = 8192
        sample_rate = 48000
        hop_size = 64
        loudness_threshold = -40
        pitch_confidence_threshold = 0.25

        filename = join(
            testdata.audio_dir, "recorded", "359500__mtg__sax-tenor-e-major.wav"
        )
        # TODO: save the below code in a function and reuse it for testARealCases

        if sys.platform == "darwin":
            import soundfile as sf

            audio, _ = sf.read(filename, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
        else:
            audio = MonoLoader(filename=filename, sampleRate=sample_rate)()
        frames = FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)

        n_frames = (audio.shape[0] - (frame_size - hop_size)) / hop_size
        step_time = hop_size / sample_rate

        # initialize audio2pitch & pitch2midi instances
        pitchDetect = Audio2Pitch(
            frameSize=frame_size,
            sampleRate=sample_rate,
            pitchConfidenceThreshold=pitch_confidence_threshold,
            loudnessThreshold=loudness_threshold,
        )

        p2m = Pitch2Midi(
            sampleRate=sample_rate,
            hopSize=hop_size,
            midiBufferDuration=0.05,
        )

        # define estimate bin and some counters
        estimated = []
        n = 0
        time_stamp = 0
        n_notes = 0

        # simulates real-time process
        for frame in frames:
            _pitch, _confidence, _loudness, _voiced = pitchDetect(frame)
            message, midi_note, time_compensation = p2m(_pitch, _voiced)
            time_stamp += step_time
            if _voiced:
                if message:
                    estimated.append(
                        [
                            n_notes,
                            time_stamp - time_compensation[1],
                            time_stamp - time_compensation[0],
                            int(midi_note[1]),
                        ]
                    )
                    # print(
                    #     f"[{n_notes}][{n}]:{(time_stamp-time_compensation[1]):.3f}, {midi2note(int(midi_note[1]))}({int(midi_note[1])})~{_pitch:.2f}Hz"  # , {time_compensation}, {midi_note}, {message}
                    # )
                    if "note_on" in message:
                        n_notes += 1
            n += 1

        ## convert note toggle messages to note features
        expected_notes = numpy.load(
            join(filedir(), "pitch2midi/359500__mtg__sax-tenor-e-major.npy")
        )

        # estimate the number of notes for expected and detected
        n_detected_notes = len(estimated)
        n_expected_notes = len(expected_notes)

        # estimate the onset error for each note and estimate the mean
        onset_mse = mean(
            [square(note[1] - estimated[int(note[0])][1]) for note in expected_notes]
        )

        # estimate the midi note error for each note and estimate the mean
        midi_note_mse = mean(
            [square(note[-1] - estimated[int(note[0])][-1]) for note in expected_notes]
        )

        # assert outputs
        self.assertAlmostEqual(onset_mse, 0, 0.01)
        self.assertAlmostEqual(n_detected_notes, n_expected_notes, 0)
        self.assertAlmostEqual(midi_note_mse, midi_note_mse, 0)

    def testARealCaseWithDMinorScale(self):
        frame_size = 8192
        sample_rate = 48000
        hop_size = 64
        loudness_threshold = -40
        pitch_confidence_threshold = 0.25
        min_frequency = 103.83
        max_frequency = 659.26

        filename = join(
            testdata.audio_dir, "recorded", "359628__mtg__sax-tenor-d-minor.wav"
        )

        if sys.platform == "darwin":
            import soundfile as sf

            audio, _ = sf.read(filename, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
        else:
            audio = MonoLoader(filename=filename, sampleRate=sample_rate)()
        frames = FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)

        n_frames = (audio.shape[0] - (frame_size - hop_size)) / hop_size
        step_time = hop_size / sample_rate

        # initialize audio2pitch & pitch2midi instances
        pitchDetect = Audio2Pitch(
            frameSize=frame_size,
            sampleRate=sample_rate,
            pitchConfidenceThreshold=pitch_confidence_threshold,
            loudnessThreshold=loudness_threshold,
            maxFrequency=max_frequency,
            minFrequency=min_frequency,
        )

        p2m = Pitch2Midi(
            sampleRate=sample_rate,
            hopSize=hop_size,
            midiBufferDuration=0.05,
        )

        # define estimate bin and some counters
        estimated = []
        n = 0
        time_stamp = 0
        n_notes = 0
        # print("Estimated notes:")

        # simulates real-time process
        for frame in frames:
            _pitch, _confidence, _loudness, _voiced = pitchDetect(frame)
            message, midi_note, time_compensation = p2m(_pitch, _voiced)
            time_stamp += step_time
            if _voiced:
                if message:
                    estimated.append(
                        [
                            n_notes,
                            time_stamp - time_compensation[1],
                            time_stamp - time_compensation[0],
                            int(midi_note[1]),
                        ]
                    )
                    # print(
                    #     f"[{n_notes}][{n}]:{(time_stamp-time_compensation[1]):.3f}, {midi2note(int(midi_note[1]))}({int(midi_note[1])})~{_pitch:.2f}Hz"  # , {time_compensation}, {midi_note}, {message}
                    # )
                    if "note_on" in message:
                        n_notes += 1
            n += 1

        ## convert note toggle messages to note features
        expected_notes = numpy.load(
            join(filedir(), f"pitch2midi/359628__mtg__sax-tenor-d-minor.npy")
        )
        # print("Expected notes:")
        # print(expected_notes)

        # estimate the number of notes for expected and detected
        n_detected_notes = len(estimated)
        n_expected_notes = len(expected_notes)

        # estimate the onset error for each note and estimate the mean
        onset_mse = mean(
            [square(note[1] - estimated[int(note[0])][1]) for note in expected_notes]
        )

        # estimate the midi note error for each note and estimate the mean
        midi_note_mse = mean(
            [square(note[-1] - estimated[int(note[0])][-1]) for note in expected_notes]
        )

        # assert outputs
        onset_tolerance = 0.01
        self.assertAlmostEqual(n_detected_notes, n_expected_notes, 0)
        self.assertAlmostEqual(onset_mse, 0, onset_tolerance)
        self.assertAlmostEqual(midi_note_mse, midi_note_mse, 0)


suite = allTests(TestPitch2Midi)

if __name__ == "__main__":
    TextTestRunner(verbosity=2).run(suite)
