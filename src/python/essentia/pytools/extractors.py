# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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
from subprocess import Popen, PIPE
from essentia import EssentiaError
import os
import sys


def __subprocess(cmd):
    """General purpose subprocess.
    """

    cmd_str = ' '.join([str(x) for x in cmd])
    print('Running "{}"...'.format(cmd_str))

    child = Popen(cmd, stdout=None, stderr=PIPE)
    _, stderr = child.communicate()
    rc = child.returncode

    return rc, cmd_str, stderr.decode("utf-8")


def __batch_extractor(audio_dir, output_dir, extractor_cmd, output_extension,
                      generate_log=True, audio_types=None, skip_analyzed=False,
                      jobs=0):
    if not audio_types:
        audio_types = ('.WAV', '.AIFF', '.FLAC', '.MP3', '.OGG')
        print("Audio files extensions considered by default: " +
              ' '.join(audio_types))
    else:
        audio_types = tuple(audio_types)
        print("Searching for audio files extensions: " + ' '.join(audio_types))
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
            if filename.upper().endswith(audio_types):
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
                if not os.path.exists(folder):
                    os.makedirs(folder)

                elif os.path.isfile(folder):
                    raise EssentiaError('Cannot create directory {}. '
                                        'There exist a file with the same name. '
                                        'Aborting analysis.'.format(folder))

                cmd_lines.append(extractor_cmd + [audio_file, out_file])

    # analyze
    log_lines = []
    errors, oks = 0, 0
    if len(cmd_lines) > 0:
        p = Pool(jobs)
        outs = p.map(__subprocess, cmd_lines)

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

    summary = "Analysis done. {} files have been skipped due to errors, {} were successfully processed and {} already existed.\n".format(
        errors, oks, skipped_count)
    print(summary)

    # generate log
    if generate_log:
        log = [summary] + log_lines

    with open(os.path.join(output_dir, 'log'), 'w') as f:
        f.write('\n'.join(log))


def batch_music_extractor(audio_dir, output_dir, generate_log=True,
                          audio_types=None, profile=None,
                          store_frames=False, skip_analyzed=False,
                          format='yaml', jobs=0):
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

    __batch_extractor(audio_dir, output_dir, extractor_cmd, 'sig', generate_log=generate_log,
                      audio_types=audio_types, skip_analyzed=skip_analyzed, jobs=jobs)


def batch_melbands_extractor(audio_dir, output_dir, generate_log=True,audio_types=None,
                             skip_analyzed=False, jobs=0, verbose=None, frame_size=None,
                             hop_size=None, number_bands=None, sample_rate=None,
                             max_frequency=None, window_type=None, compression_type=None,
                             normalize=None):
    """Generates mel bands for every audio file matching `audio_types` in `audio_dir`.
    The generated .npy files are stored in `output_dir` matching the folder
    structure found in `audio_dir`.
    """

    extractor_cmd = [sys.executable, os.path.join(os.path.dirname(__file__),
                                                  'extractors/melbands_extractor.py')]

    if verbose:
        extractor_cmd += ['--verbose']

    if normalize:
        extractor_cmd += ['--normalize']

    if frame_size:
        extractor_cmd += ['--frame_size', str(frame_size)]

    if hop_size:
        extractor_cmd += ['--hop_size', str(hop_size)]

    if number_bands:
        extractor_cmd += ['--number_bands', str(number_bands)]

    if sample_rate:
        extractor_cmd += ['--sample_rate', str(sample_rate)]

    if max_frequency:
        extractor_cmd += ['--max_frequency', str(max_frequency)]

    if window_type:
        extractor_cmd += ['--window_type', str(window_type)]

    if compression_type:
        extractor_cmd += ['--compression_type', compression_type]


    __batch_extractor(audio_dir, output_dir, extractor_cmd, 'npy',
                      generate_log=generate_log, audio_types=audio_types,
                      skip_analyzed=skip_analyzed, jobs=jobs)
