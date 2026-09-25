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


# Tests for AudioLoader on files with more than one stream (issue MTG#325).
#
# AudioLoader::process() reads packets in a loop and discards every packet
# that does not belong to the selected audio stream. These tests exercise that
# discard path in two ways:
#
#   * functionally: stream selection on a file with two audio streams, and
#     decoding the audio of a video+audio file (every video packet goes
#     through the discard path), and
#   * as a leak regression: before the fix the discarded packets' payloads
#     were never unref'd and leaked for the lifetime of the process, so
#     repeatedly loading a multi-stream file made peak RSS grow linearly.
#
# The multi-stream fixtures cannot live in the test/audio submodule from this
# repository, so they are synthesised on the fly with the ffmpeg CLI; the
# whole suite is skipped when ffmpeg is not available.

from essentia_test import *
from essentia.standard import AudioLoader as stdAudioLoader
from essentia.standard import MonoLoader as stdMonoLoader
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest


SAMPLERATE = 44100
DURATION = 3  # seconds


def generateFixtures(directory):
    """Generate the multi-stream fixtures with the ffmpeg CLI.

    Returns (twoAudioFile, audioVideoFile). Raises on encoder/muxer failure.
    """

    def ffmpeg(*args):
        subprocess.run(['ffmpeg', '-y', '-nostdin', '-loglevel', 'error'] + list(args),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sine = 'sine=frequency=%d:sample_rate=%d:duration=%d' % (440, SAMPLERATE, DURATION)
    sine880 = 'sine=frequency=%d:sample_rate=%d:duration=%d' % (880, SAMPLERATE, DURATION)

    # two audio streams (440 Hz and 880 Hz sines) in one Matroska container
    twoAudioFile = join(directory, 'two_streams.mka')
    ffmpeg('-f', 'lavfi', '-i', sine,
           '-f', 'lavfi', '-i', sine880,
           '-map', '0:a', '-map', '1:a', '-c:a', 'aac', '-b:a', '128k',
           twoAudioFile)

    # video + audio: every video packet goes through the loader's discard
    # path. The video track is stored as rawvideo so its packet payload is
    # large and deterministic (320x240 yuv420p @ 25 fps = ~8.6 MB of video
    # payload per load), making a leak of discarded packets unambiguous.
    audioVideoFile = join(directory, 'video_audio.mkv')
    ffmpeg('-f', 'lavfi', '-i', 'testsrc=duration=%d:size=320x240:rate=25' % DURATION,
           '-f', 'lavfi', '-i', sine,
           '-map', '0:v', '-map', '1:a',
           '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
           audioVideoFile)

    return twoAudioFile, audioVideoFile


@unittest.skipIf(shutil.which('ffmpeg') is None,
                 'the ffmpeg CLI is required to generate multi-stream fixtures')
class TestAudioLoader_MultiStream(TestCase):

    tmpdir = None
    twoAudioFile = None
    audioVideoFile = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='essentia_multistream_')
        try:
            cls.twoAudioFile, cls.audioVideoFile = generateFixtures(cls.tmpdir)
        except (OSError, subprocess.CalledProcessError):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
            cls.tmpdir = None
            raise unittest.SkipTest('the local ffmpeg cannot generate the multi-stream fixtures')

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir:
            shutil.rmtree(cls.tmpdir, ignore_errors=True)


    def dominantFrequency(self, audio):
        # frequency of the strongest spectral peak of a mono signal
        spectrum = numpy.abs(numpy.fft.rfft(numpy.asarray(audio, dtype=numpy.float64)))
        spectrum[0] = 0  # ignore DC
        return numpy.argmax(spectrum) * SAMPLERATE / float(len(audio))

    def assertDecodedSine(self, audio, frequency):
        # ~3 s of audio (AAC adds/removes up to a couple thousand samples of
        # encoder delay/padding at the edges)
        self.assertTrue(abs(len(audio) - DURATION * SAMPLERATE) < 0.15 * SAMPLERATE,
                        'expected ~%d decoded samples, got %d' % (DURATION * SAMPLERATE, len(audio)))
        found = self.dominantFrequency(audio)
        self.assertTrue(abs(found - frequency) < 5,
                        'expected a dominant spectral peak at %d Hz, found %.1f Hz' % (frequency, found))


    # ---- functional: the discard path must skip foreign packets correctly ----

    def testTwoAudioStreamsAudioLoader(self):
        # audioStream selects which stream is decoded; packets of the other
        # stream are discarded
        for stream, frequency in [(0, 440), (1, 880)]:
            audio, sr, channels, _, _, _ = stdAudioLoader(filename=self.twoAudioFile,
                                                          audioStream=stream)()
            self.assertEqual(sr, SAMPLERATE)
            mono = numpy.asarray(audio).mean(axis=1)
            self.assertDecodedSine(mono, frequency)

    def testTwoAudioStreamsMonoLoader(self):
        for stream, frequency in [(0, 440), (1, 880)]:
            audio = stdMonoLoader(filename=self.twoAudioFile,
                                  sampleRate=SAMPLERATE, audioStream=stream)()
            self.assertDecodedSine(audio, frequency)

    def testVideoPlusAudio(self):
        # all the video packets go through the discard path; the audio must
        # still decode completely and correctly
        audio, sr, channels, _, _, _ = stdAudioLoader(filename=self.audioVideoFile)()
        self.assertEqual(sr, SAMPLERATE)
        mono = numpy.asarray(audio).mean(axis=1)
        self.assertDecodedSine(mono, 440)


    # ---- regression: discarded packets must not leak (issue MTG#325) ----

    def peakRSSBytes(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in bytes on macOS, kilobytes on Linux
        return rss if sys.platform == 'darwin' else rss * 1024

    def testDiscardedPacketsDoNotLeak(self):
        # Before the fix every discarded packet's payload leaked, so each load
        # of the video+audio fixture leaked its whole video stream (~8.6 MB):
        # 15 measured loads leaked >100 MB. After the fix, repeated loads
        # reuse already-peaked memory and RSS stays flat, so a generous
        # threshold separates the two cases without flakiness.
        def load():
            audio, _, _, _, _, _ = stdAudioLoader(filename=self.audioVideoFile)()
            return len(audio)

        # warm up: let allocator pools, codec tables etc. reach their
        # high-water mark before taking the baseline
        for _ in range(5):
            load()
        baseline = self.peakRSSBytes()

        iterations = 15
        for _ in range(iterations):
            load()
        growth = self.peakRSSBytes() - baseline

        limit = 16 * 1024 * 1024  # ~8x below the pre-fix leak over 15 loads
        self.assertTrue(growth < limit,
                        'peak RSS grew by %.1f MB over %d loads of a multi-stream file '
                        '(limit %.1f MB): discarded packets are probably leaking'
                        % (growth / 1048576.0, iterations, limit / 1048576.0))


suite = allTests(TestAudioLoader_MultiStream)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
