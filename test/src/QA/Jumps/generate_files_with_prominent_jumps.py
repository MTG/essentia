#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Created by pablo on 2/02/18

"""
This script find all the files in a folder, tries to open them as audio files,
removes some samples in order to generate a jump and ens and saves them in a desired folder
"""

import numpy as np
import os
import essentia.standard as es
from essentia import array as esarr


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename.lower().endswith(pattern):
                filename = os.path.join(root, basename)
                yield filename


if __name__ == '__main__':
    in_folder = '../../audio/recorded'
    out_folder = '../../QA-audio/Jumps/loud_songs'
    fs = 44100.
    """
    files = [x for x in find_files(in_folder, 'wav')]

    for f in files:
        try:
            audio = es.MonoLoader(filename=f, sampleRate=fs)()
        except Exception:
            print '{} was not loaded'.format(f)
            continue
    """
    f = '../../../../../../pablo/Music/Desakato-La_Teoria_del_Fuego/03. Desakato - Estigma.mp3'
    audio = es.MonoLoader(filename=f, sampleRate=fs)()
    original_len = len(audio)

    start_jump = original_len/4
    if audio[start_jump] > 0:
        end = next(idx for idx, i in enumerate(audio[start_jump:]) if i < -.3) # Want at least a gap of .5
    else:
        end = next(idx for idx, i in enumerate(audio[start_jump:]) if i > .3)

    end_jump = start_jump + end

    audio = np.hstack([audio[:start_jump], audio[end_jump:]])

    text = ['{}\t0.0\tevent\n'.format(start_jump/float(fs))]

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    f_name = ''.join(os.path.basename(f).split('.')[:-1])
    with open('{}/{}_prominent_jump.lab'.format(out_folder, f_name), 'w') as o_file:
        o_file.write(''.join(text))

    es.MonoWriter(filename='{}/{}_prominent_jump.wav'.format(out_folder, f_name))(esarr(audio))
