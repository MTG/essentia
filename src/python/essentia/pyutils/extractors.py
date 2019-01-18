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
from subprocess import Popen, PIPE

import numpy as np
import os
import sys
import fnmatch


def __subprocess__(cmd):
    """
    General purpose subprocess.
    """

    cmd_str = ' '.join(cmd)
    print('Running "{}"...'.format(cmd_str))

    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE).communicate()

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    stderr = 'cmd: "{}"\nstderr:"{}"'.format(cmd_str, stderr)
    stdout = 'cmd: "{}"\nstderr:"{}"'.format(cmd_str, stdout)
    return stdout, stderr


def recursive_music_extractor(audio_dir, output_dir, generate_log=True,
                              audio_types=None, profile=None, 
                              store_frames=False, skip_analyzed=False, jobs=4):
    """
        print('You cannot specify the same descriptor patterns in both 'include_descs' and --ignore flags')
        sys.exit() # TODO return None instead in this function
    """

    if not audio_types:
        audio_types = ['*.wav', '*.aiff', '*.flac', '*.mp3', '*.ogg']
        print("Audio files extensions considered by default: " + ' '.join(audio_types))
    else:
        print("Searching for audio files extensions: " + ' '.join(audio_types))
    print("")

    output_dir = os.path.abspath(output_dir)

    if profile:
        assert os.path.isfile(profile)

    # find all audio files and prepare folder structure in the output folder
    os.chdir(audio_dir)

    skipped_count = 0
    skipped_files = []
    cmd_lines = []
    for root, dirnames, filenames in os.walk("."):
        for match in audio_types:
            for filename in fnmatch.filter(filenames, match):
                audio_file = os.path.relpath(os.path.join(root, filename))
                audio_file_abs = os.path.join(audio_dir, audio_file)
                sig_file = os.path.join(output_dir, audio_file + ".sig")

                if skip_analyzed:
                    if os.path.isfile(sig_file):
                        print("Found descriptor file for " + audio_file + ", skipping...")
                        skipped_files.append(audio_file)
                        skipped_count += 1
                        continue

                folder = os.path.dirname(sig_file)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                elif os.path.isfile(folder):
                    print("Cannot create directory %s" % folder)
                    print("There exist a file with the same name. Aborting analysis.")
                    sys.exit()

                # TODO: music_extractor.py path could be obtained in a better way
                cmd_lines.append([sys.executable, os.path.join(os.path.dirname(__file__),
                                  'extractors/music_extractor.py')] + [audio_file_abs, sig_file])

    # analyze
    errors, oks = 0, 0
    if len(cmd_lines) > 0:
        p = Pool(jobs)
        outs = p.map(__subprocess__, cmd_lines)

        stdout, stderr = zip(*outs)

        stderr = list(stderr)
        stdout = list(stdout)

        status = np.array(["ok!" in it for it in stderr])

        errors = np.count_nonzero(status == False)
        oks = np.count_nonzero(status == True)

    summary = "Analysis done. {} files have been skipped due to errors, {} were processed and {} already existed.".format(errors, oks, skipped_count)
    print(summary)

    # generate log
    if generate_log:
        log = [summary]

        if errors > 0:
            log += ['Errors:'] + [stderr[idx] for idx, i in enumerate(status) if not i]

        if oks > 0:
            log += ['Oks:'] + [stderr[idx] for idx, i in enumerate(status) if i]

        if skipped_count > 0:
            log += ['Skipped files:'] + skipped_files

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'log'), 'w') as f:
            f.write('\n'.join(log))
