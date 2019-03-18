#!/usr/bin/env python

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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


import os

import numpy as np

import essentia.standard as es
from essentia import array as esarr

PI = np.pi


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename.lower().endswith(pattern):
                filename = os.path.join(root, basename)
                yield filename


if __name__ == '__main__':
    """
    This script find all the files in a folder, tries to open them as audio files,
    reduces a bit the audio level to prevent clipping and ads a humming tone at 50 Hz
    """

    in_folder = '/home/pablo/data/sns-small/samples'
    out_folder = '/home/pablo/reps/essentia/test/QA-audio/Hum/Songs50HzHum'
    fs = 44100.
    files = [x for x in find_files(in_folder, 'flac')]
    if not files:
        print('no files found!')

    for f in files:
        try:
            audio = es.MonoLoader(filename=f, sampleRate=fs)()
        except Exception:
            print('{} was not loaded'.format(f))
            continue

        fs = 44100.
        t = np.linspace(0, len(audio) / fs, len(audio))

        freq = 50

        sinusoid = np.sin(2 * PI * freq * t)

        signal = np.array(.95 * audio + .005 * sinusoid, dtype=np.float32)

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        f_name = ''.join(os.path.basename(f).split('.')[:-1])

        es.MonoWriter(filename='{}/{}_hum.wav'.format(out_folder, f_name))(esarr(signal))
