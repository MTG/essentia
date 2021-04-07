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

from argparse import ArgumentParser
from multiprocessing import Pool
from multiprocessing import cpu_count
from subprocess import run, PIPE
from essentia import EssentiaError
from functools import partial
import os
import sys


def _subprocess(cmd, verbose=True):
    """General purpose subprocess."""

    completed_process = run(cmd, stdout=None, stderr=PIPE)

    cmd_str = ' '.join([str(x) for x in cmd])
    stderr = completed_process.stderr.decode("utf-8")
    rc = completed_process.returncode

    if verbose:
        if rc == 0:
            print('"{}"... ok!'.format(cmd_str))
        else:
            print('"{}"... failed (returncode {})!'.format(cmd_str, rc))
            print(stderr, '\n')

    return rc, cmd_str, stderr


def _batch_extractor(audio_dir, output_dir, extractor_cmd, output_extension,
                     generate_log=True, audio_types=None, skip_analyzed=False,
                     jobs=0, verbose=True):
    if not audio_types:
        audio_types = ('.wav', '.aiff', '.flac', '.mp3', '.ogg')
        print("Audio files extensions considered by default: " +
              ', '.join(audio_types))
    else:
        if type(audio_types) == str:
            audio_types = [audio_types]

        audio_types = tuple(audio_types)
        audio_types = tuple([i.lower() for i in audio_types])
        print("Searching for audio files extensions: " + ', '.join(audio_types))
    print("")

    output_dir = os.path.abspath(output_dir)
    audio_dir = os.path.abspath(audio_dir)

    if jobs == 0:
        try:
            jobs = cpu_count()
        except NotImplementedError:
            print('Failed to automatically detect the cpu count, '
                  'the analysis will try to continue with 4 jobs. '
                  'For a different behavior change the `job` parameter.')
            jobs = 4

    skipped_count = 0
    skipped_files = []
    cmd_lines = []
    for root, _, filenames in os.walk(audio_dir):
        for filename in filenames:
            if filename.lower().endswith(audio_types):
                audio_file = os.path.join(audio_dir, root, filename)
                out_file = os.path.join(output_dir, output_dir, filename)

                if skip_analyzed:
                    if os.path.isfile( '{}.{}'.format(out_file, output_extension)):
                        print("Found descriptor file for " +
                              audio_file + ", skipping...")
                        skipped_files.append(audio_file)
                        skipped_count += 1
                        continue
                folder = os.path.dirname(out_file)

                if os.path.isfile(folder):
                    raise EssentiaError('Cannot create directory "{}". '
                                        'There is a file with the same name. '
                                        'Aborting analysis.'.format(folder))
                else:
                    os.makedirs(folder, exist_ok=True)

                cmd_lines.append(extractor_cmd + [audio_file, out_file])

    # analyze
    log_lines = []
    total, errors, oks = 0, 0, 0
    if cmd_lines:
        p = Pool(jobs)
        outs = p.map(partial(_subprocess, verbose=verbose), cmd_lines)

        total = len(outs)
        status, cmd, stderr = zip(*outs)

        oks, errors = 0, 0
        for i, cmd_idx, err in zip(status, cmd, stderr):
            if i == 0:
                oks += 1
                log_lines.append('"{}" ok!'.format(cmd_idx))
            else:
                errors += 1
                log_lines.append('"{}" failed'.format(cmd_idx))
                log_lines.append('  "{}"'.format(err))

    summary = ("Analysis done for {} files. {} files have been skipped due to errors, "
               "{} were successfully processed and {} already existed.\n").format(total, errors, oks, skipped_count)
    print(summary)

    # generate log
    if generate_log:
        log = [summary] + log_lines

        with open(os.path.join(output_dir, 'log'), 'w') as f:
            f.write('\n'.join(log))


def batch_music_extractor(audio_dir, output_dir, generate_log=True, audio_types=None, profile=None,
                          store_frames=False, skip_analyzed=False, format='yaml', jobs=0):
    """Processes every audio file matching `audio_types` in `audio_dir` with MusicExtractor.
    The generated .sig yaml/json files are stored in `output_dir` matching the folder
    structure found in `audio_dir`.
    """

    extractor_cmd = [sys.executable, os.path.join(os.path.dirname(__file__),
                                                  'extractors/music_extractor.py'),
                     '--format', format]

    if profile:
        assert os.path.isfile(profile)
        extractor_cmd += ['--profile', profile]

    if store_frames:
        extractor_cmd += ['--store_frames']

    _batch_extractor(audio_dir, output_dir, extractor_cmd, 'sig', generate_log=generate_log,
                     audio_types=audio_types, skip_analyzed=skip_analyzed, jobs=jobs)


def batch_melspectrogram(audio_dir, output_dir, generate_log=True, verbose=True, audio_types=None,
                         skip_analyzed=True, jobs=0, sample_rate=None, frame_size=None, hop_size=None,
                         window_type=None, zero_padding=None, low_frequency_bound=None,
                         high_frequency_bound=None, number_bands=None, warping_formula=None,
                         weighting=None, normalize=None, bands_type=None, compression_type=None):
    """Generates mel bands for every audio file matching `audio_types` in `audio_dir`.
    The generated .npy files are stored in `output_dir` matching the folder
    structure found in `audio_dir`.
    """

    extractor_cmd = [sys.executable, os.path.join(os.path.dirname(__file__),
                                                  'extractors/melspectrogram.py')]

    # Set --force as a hardcoded flat.
    # Use skip_analyzed to control this behavior.
    extractor_cmd += ['--force']

    if verbose:
        extractor_cmd += ['--verbose']

    if sample_rate:
        extractor_cmd += ['--sample-rate', str(sample_rate)]

    if frame_size:
        extractor_cmd += ['--frame-size', str(frame_size)]

    if hop_size:
        extractor_cmd += ['--hop-size', str(hop_size)]

    if window_type:
        extractor_cmd += ['--window-type', str(window_type)]

    if zero_padding:
        extractor_cmd += ['--zero-padding', str(zero_padding)]

    if low_frequency_bound:
        extractor_cmd += ['--low-frequency-bound', str(low_frequency_bound)]

    if high_frequency_bound:
        extractor_cmd += ['--high-frequency-bound', str(high_frequency_bound)]

    if number_bands:
        extractor_cmd += ['--number-bands', str(number_bands)]

    if warping_formula:
        extractor_cmd += ['--warping-formula', str(warping_formula)]

    if weighting:
        extractor_cmd += ['--weighting', str(weighting)]

    if normalize:
        extractor_cmd += ['--normalize', str(normalize)]

    if bands_type:
        extractor_cmd += ['--bands-type', str(bands_type)]

    if compression_type:
        extractor_cmd += ['--compression-type', str(compression_type)]

    _batch_extractor(audio_dir, output_dir, extractor_cmd, 'npy',
                     generate_log=generate_log, audio_types=audio_types,
                     skip_analyzed=skip_analyzed, jobs=jobs, verbose=verbose)
