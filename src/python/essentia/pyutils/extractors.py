# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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


def __subprocess__(cmd):
    """
    General purpose subprocess.
    """

    cmd_str = ' '.join(cmd)
    print('Running "{}"...'.format(cmd_str))

    child = Popen(cmd, stdout=None, stderr=PIPE)
    _, stderr = child.communicate()
    rc = child.returncode

    return rc, cmd_str, stderr.decode("utf-8")


def batch_music_extractor(audio_dir, output_dir, generate_log=True,
                          audio_types=None, profile=None,
                          store_frames=False, skip_analyzed=False,
                          format='yaml', jobs=0):
    """
        Processes every audio file matching `audio_types` in `audio_dir` with MusicExtractor. The generated
        .sig yaml/json files are stored in `output_dir` matching the folder structure found in `audio_dir`.
    """

    if not audio_types:
        audio_types = ('.WAV', '.AIFF', '.FLAC', '.MP3', '.OGG')
        print("Audio files extensions considered by default: " +
              ' '.join(audio_types))
    else:
        audio_types = tuple(audio_types)
        print("Searching for audio files extensions: " + ' '.join(audio_types))
    print("")

    output_dir = os.path.abspath(output_dir)

    if profile:
        assert os.path.isfile(profile)

    if jobs == 0:
        try:
            jobs = cpu_count()
        except NotImplementedError:
            print("Failed to automatically detect the cpu count, the analysis will try to continue with 4 jobs. For a different behavior change the `job` parameter.")
            jobs = 4

    # find all audio files and prepare folder structure in the output folder
    os.chdir(audio_dir)

    skipped_count = 0
    skipped_files = []
    cmd_lines = []
    for root, dirnames, filenames in os.walk("."):
        for filename in filenames:
            if filename.upper().endswith(audio_types):
                audio_file = os.path.relpath(os.path.join(root, filename))
                audio_file_abs = os.path.join(audio_dir, audio_file)
                sig_file = os.path.join(output_dir, audio_file)

                if skip_analyzed:
                    if os.path.isfile(sig_file + '.sig'):
                        print("Found descriptor file for " +
                              audio_file + ", skipping...")
                        skipped_files.append(audio_file)
                        skipped_count += 1
                        continue
                folder = os.path.dirname(sig_file)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                elif os.path.isfile(folder):
                    raise EssentiaError('Cannot create directory {} .There exist a file with the same name. Aborting analysis.'.format(folder))

                cmd_line = [
                            sys.executable,
                            os.path.join(os.path.dirname(__file__), 'extractors/music_extractor.py'), 
                            audio_file_abs, sig_file,
                            '--format', format
                            ]

                if store_frames:
                    cmd_line += ['--store_frames']

                if profile:
                    cmd_line += ['--profile', profile]

                cmd_lines.append(cmd_line)


    # analyze
    errors, oks = 0, 0
    if len(cmd_lines) > 0:
        p = Pool(jobs)
        outs = p.map(__subprocess__, cmd_lines)

        status, cmd, stderr = zip(*outs)

        oks, errors = 0, 0
        for i in status:
            if i == 0:
                oks += 1
            else:
                errors += 1

    summary = "Analysis done. {} files have been skipped due to errors, {} were processed and {} already existed.".format(
        errors, oks, skipped_count)
    print(summary)

    # generate log
    if generate_log:
        log = [summary]

        if errors > 0:
            log += ['Errors:'] + ['"{}"\n{}\n\n'.format(cmd[idx], stderr[idx])
                                  for idx, i in enumerate(status) if i != 0]

        if oks > 0:
            log += ['Oks:'] + [cmd[idx]
                               for idx, i in enumerate(status) if i == 0]

        if skipped_count > 0:
            log += ['Skipped files:'] + skipped_files

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'log'), 'w') as f:
            f.write('\n'.join(log))
