#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Created by pablo on 2/02/18

"""
This script find all the files in a folder, tries to open them as audio files,
reduces a bit the audio level to prevent clipping and ads a humming tone at 50 Hz
"""

import numpy as np
import os
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
            print '{} was not loaded'.format(f)
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
