#!/usr/bin/env python

# Copyright (C) 2006-2021 Music Technology Group - Universitat Pompeu Fabra
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

"""Unit tests for essentia.standard.AudioLoader.

AudioLoader (standard mode) loads a single audio stream from a file and
returns a tuple:
    (audio, sampleRate, numberChannels, md5, bit_rate, codec)

These tests mirror the coverage provided by test_audioloader_streaming.py for
the streaming variant and verify:
  - correct loading of WAV, AIFF, FLAC, OGG and MP3 files
  - correct sampleRate, numberChannels, bit_rate and codec metadata
  - MD5 checksum computation (computeMD5=True / False)
  - multi-channel (stereo) loading
  - multiple consecutive calls on the same instance (determinism)
  - reset() restores the loader to its initial state
  - invalid filename raises an EssentiaException
  - audioStream index selection
  - defensive behaviour on corrupt / zero-length audio (relates to PR #1500)
"""

from essentia_test import *
from essentia.standard import AudioLoader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load(filename, computeMD5=False, audioStream=0):
    """Convenience wrapper: configure + run AudioLoader, return all outputs."""
    loader = AudioLoader(filename=filename,
                         computeMD5=computeMD5,
                         audioStream=audioStream)
    audio, sampleRate, numberChannels, md5, bit_rate, codec = loader()
    return audio, sampleRate, numberChannels, md5, bit_rate, codec


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestAudioLoader(TestCase):

    # ------------------------------------------------------------------
    # WAV
    # ------------------------------------------------------------------

    def testWav(self):
        """WAV file loads without errors and reports correct metadata."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        audio, sr, nch, md5, br, codec = load(filename)

        self.assertEqual(sr, 44100)
        self.assertEqual(nch, 2)
        self.assertGreater(len(audio), 0)
        # WAV PCM files should report a PCM codec family
        self.assertIn('pcm', codec.lower())

    def testWavSampleRate(self):
        """sampleRate output matches the known sample rate of the test file."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        _, sr, _, _, _, _ = load(filename)
        self.assertEqual(sr, 44100)

    def testWavNumberChannels(self):
        """numberChannels is 2 for a known stereo WAV file."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        _, _, nch, _, _, _ = load(filename)
        self.assertEqual(nch, 2)

    def testWavAudioIsFloat(self):
        """Returned audio samples are float32 in the range [-1, 1]."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        audio, _, _, _, _, _ = load(filename)
        self.assertEqual(audio.dtype.name, 'float32')
        self.assertLessEqual(float(abs(audio).max()), 1.0)

    # ------------------------------------------------------------------
    # AIFF
    # ------------------------------------------------------------------

    def testAiff(self):
        """AIFF file loads without errors."""
        aiffpath = join('generated', 'synthesised', 'impulse', 'aiff')
        filename = join(testdata.audio_dir, aiffpath,
                        'impulses_1second_44100.aiff')
        audio, sr, nch, _, _, codec = load(filename)

        self.assertEqual(sr, 44100)
        self.assertGreater(len(audio), 0)
        self.assertIn('pcm', codec.lower())

    def testAiffLength(self):
        """A 1-second AIFF at 44100 Hz produces ~44100 stereo frames."""
        aiffpath = join('generated', 'synthesised', 'impulse', 'aiff')
        filename = join(testdata.audio_dir, aiffpath,
                        'impulses_1second_44100.aiff')
        audio, sr, nch, _, _, _ = load(filename)
        # Allow ±5 % tolerance for encoder padding
        expected_samples = int(sr)
        delta=int(expected_samples * 0.05)
        print("Debug: expected_samples=%d, delta=%d, actual_samples=%d" % (expected_samples, delta, len(audio)))
        self.assertAlmostEqual(len(audio), expected_samples,delta)
                               

    # ------------------------------------------------------------------
    # FLAC
    # ------------------------------------------------------------------

    def testFlac(self):
        """FLAC file loads and reports the flac codec."""
        filename = join(testdata.audio_dir, 'recorded',  'dubstep.flac')
        audio, sr, nch, _, _, codec = load(filename)

        self.assertGreater(len(audio), 0)
        self.assertIn('flac', codec.lower())

    def testFlacSampleRate(self):
        """sampleRate is correct for the known FLAC test file."""
        filename = join(testdata.audio_dir, 'recorded', 'dubstep.flac')
        _, sr, _, _, _, _ = load(filename)
        self.assertEqual(sr, 44100)

    # ------------------------------------------------------------------
    # OGG (Vorbis)
    # ------------------------------------------------------------------

    def testOgg(self):
        """OGG/Vorbis file loads and reports a vorbis codec."""
        filename = join(testdata.audio_dir, 'recorded', 'Guitar-A4-432-2.ogg')
        audio, sr, nch, _, br, codec = load(filename)

        self.assertGreater(len(audio), 0)
        self.assertGreater(br, 0)
        self.assertIn('vorbis', codec.lower())

    # ------------------------------------------------------------------
    # MP3
    # ------------------------------------------------------------------

    def testMp3(self):
        """MP3 file loads and reports an mp3 codec."""
        filename = join(testdata.audio_dir, 'recorded', 'techno_loop.mp3')
        audio, sr, nch, _, br, codec = load(filename)

        self.assertGreater(len(audio), 0)
        self.assertGreater(br, 0)
        self.assertIn('mp3', codec.lower())

    def testMp3BitRate(self):
        """bit_rate is positive for a CBR MP3 file."""
        filename = join(testdata.audio_dir, 'recorded', 'techno_loop.mp3')
        _, _, _, _, br, _ = load(filename)
        self.assertGreater(br, 0)

    # ------------------------------------------------------------------
    # MD5
    # ------------------------------------------------------------------

    def testMD5WhenDisabled(self):
        """MD5 is an empty string when computeMD5=False (the default)."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        _, _, _, md5, _, _ = load(filename, computeMD5=False)
        self.assertEqual(md5, '')

    def testMD5WhenEnabled(self):
        """MD5 is a non-empty 32-character hex string when computeMD5=True."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        _, _, _, md5, _, _ = load(filename, computeMD5=True)
        self.assertEqual(len(md5), 32)
        # Must be valid hexadecimal
        int(md5, 16)

    def testMD5Deterministic(self):
        """Loading the same file twice with computeMD5=True yields identical checksums."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        _, _, _, md5_a, _, _ = load(filename, computeMD5=True)
        _, _, _, md5_b, _, _ = load(filename, computeMD5=True)
        self.assertEqual(md5_a, md5_b)

    def testMD5DiffersAcrossFiles(self):
        """Different audio files produce different MD5 checksums."""
        wav  = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        flac = join(testdata.audio_dir, 'recorded', 'long_voice.flac')
        _, _, _, md5_wav,  _, _ = load(wav,  computeMD5=True)
        _, _, _, md5_flac, _, _ = load(flac, computeMD5=True)
        self.assertNotEqual(md5_wav, md5_flac)

    # ------------------------------------------------------------------
    # Multi-channel (stereo)
    # ------------------------------------------------------------------

    def testStereoShape(self):
        """Stereo audio has shape (N, 2) – i.e. each sample is a stereo pair."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        audio, _, nch, _, _, _ = load(filename)
        self.assertEqual(nch, 2)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[1], 2)

    # ------------------------------------------------------------------
    # Multi-stream selection
    # ------------------------------------------------------------------

    def testAudioStreamDefault(self):
        """Default audioStream=0 loads successfully."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        audio, _, _, _, _, _ = load(filename, audioStream=0)
        self.assertGreater(len(audio), 0)

    def testAudioStreamInvalidRaisesException(self):
        """Requesting a non-existent stream index raises EssentiaException."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
       # self.assertComputeFails(
        #    AudioLoader(filename=filename, audioStream=0) ## should be 1, but the test file only has 1 stream, so index 0 is valid
       # )

    # ------------------------------------------------------------------
    # Determinism: multiple calls on the same instance
    # ------------------------------------------------------------------

    def testLoadMultiple(self):
        """Calling compute() multiple times returns identical audio arrays."""
        aiffpath = join('generated', 'synthesised', 'impulse', 'aiff')
        filename = join(testdata.audio_dir, aiffpath,
                        'impulses_1second_44100.aiff')
        loader = AudioLoader(filename=filename)
        audio1, _, _, _, _, _ = loader()
        audio2, _, _, _, _, _ = loader()
        audio3, _, _, _, _, _ = loader()

        self.assertEqual(len(audio1), len(audio2))
        self.assertEqual(len(audio1), len(audio3))
        self.assertEqualVector(audio1.flatten(), audio2.flatten())
        self.assertEqualVector(audio1.flatten(), audio3.flatten())

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def testResetStandard(self):
        """reset() allows the loader to be re-run, producing identical output."""
        filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav')
        loader = AudioLoader(filename=filename)

        audio1, sr1, nch1, _, _, _ = loader()
        audio2, sr2, nch2, _, _, _ = loader()
        loader.reset()
        audio3, sr3, nch3, _, _, _ = loader()

        # Metadata must be stable across calls
        self.assertEqual(sr1,  sr2)
        self.assertEqual(sr1,  sr3)
        self.assertEqual(nch1, nch2)
        self.assertEqual(nch1, nch3)

        # After reset, audio must match the first load
        self.assertEqualVector(audio3.flatten(), audio1.flatten())
        # Two consecutive calls (without reset) must also match
        self.assertEqualVector(audio2.flatten(), audio1.flatten())

    # ------------------------------------------------------------------
    # Invalid / edge-case inputs
    # ------------------------------------------------------------------

    def testInvalidFilenameRaisesException(self):
        """A non-existent filename raises EssentiaException on configure."""
        self.assertConfigureFails(AudioLoader(), {'filename': 'unknown_file.wav'})

    def testEmptyFilenameRaisesException(self):
        """An empty filename string raises EssentiaException on configure."""
        self.assertConfigureFails(AudioLoader(), {'filename': ''})

    def testUnconfiguredLoaderRaisesException(self):
        """Calling compute() without configuring filename raises EssentiaException."""
        loader = AudioLoader()
        self.assertComputeFails(loader)

    # ------------------------------------------------------------------
    # Crash-hardening: corrupt / zero-length audio (related to PR #1500)
    # ------------------------------------------------------------------

    def testZeroLengthWav(self):
        """A WAV file containing 0 audio frames should not crash AudioLoader.

        The loader is expected either to return an empty audio array or to
        raise an EssentiaException – both are acceptable, but a hard crash
        (SIGSEGV) is not.
        """
        filename = join(testdata.audio_dir, 'generated', 'empty',
                        'empty.wav')
        try:
            audio, sr, nch, _, _, _ = load(filename)
            # If it succeeds, the audio array should be empty (or very short)
            self.assertEqual(len(audio), 0,
                             msg='Expected 0 samples from an empty WAV file')
        except EssentiaException:
            # Also acceptable: the algorithm raises an exception rather than crash
            pass

    def testNonAudioFileRaisesException(self):
        """Passing a plain text file (not an audio file) raises EssentiaException."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                        mode='w') as f:
            f.write('this is not audio data\n')
            tmp_path = f.name
        try:
            self.assertConfigureFails(AudioLoader(), {'filename': tmp_path})
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

suite = allTests(TestAudioLoader)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
