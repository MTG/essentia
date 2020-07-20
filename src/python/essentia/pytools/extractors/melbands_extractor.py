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

import argparse
import os

import numpy as np

from essentia import Pool
from essentia import run

from essentia.streaming import (MonoLoader, FrameCutter, Windowing, Spectrum,
                                MelBands, UnaryOperator)


def melbands_extractor(args):
    filename_in = args.audio_file
    filename_out = args.npy_file
    verbose = args.verbose
    force = args.force

    frame_size = args.frame_size
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    min_frequency = args.min_frequency
    max_frequency = args.max_frequency
    window_type = args.window_type
    number_bands = args.number_bands
    compression_type = args.compression_type

    spectrum_size = frame_size // 2 + 1

    if not force:
        if os.path.exists(filename_out):
            if verbose:
                print('Skipping "{}"'.format(filename_out))
            return

    pool = Pool()

    loader = MonoLoader(filename=filename_in,
                        sampleRate=sample_rate)
    frameCutter = FrameCutter(frameSize=frame_size,
                              hopSize=hop_size)
    w = Windowing(type=window_type,
                  normalized=False)
    spec = Spectrum(size=frame_size)
    mels = MelBands(inputSize=spectrum_size,
                    sampleRate=sample_rate,
                    numberBands=number_bands,
                    weighting='linear',
                    normalize='unit_tri',
                    lowFrequencyBound=min_frequency,
                    highFrequencyBound=max_frequency)

    if compression_type.lower() == 'db':
        shift = UnaryOperator(type='identity')
        compresssor = UnaryOperator(type='lin2db')

    elif compression_type.lower() == 'log':
        shift = UnaryOperator(type='identity', scale=1e4, shift=1)
        compresssor = UnaryOperator(type='log')

    loader.audio >> frameCutter.signal
    frameCutter.frame >> w.frame >> spec.frame
    spec.spectrum >> mels.spectrum
    mels.bands >> shift.array >> compresssor.array >> (pool, 'mel_bands')

    run(loader)

    mel_bands = np.array(pool['mel_bands'])

    np.save(filename_out, mel_bands)

    if verbose:
        print('Done for "{}"'.format(filename_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Computes mel bands and stores them as a numpy binary file.')

    parser.add_argument('audio_file',
                        help='audio file name')
    parser.add_argument('npy_file',
                        help='output .npy file name')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='whether to print out status to the standard output')
    parser.add_argument('--force', '-f', action='store_true',
                        help='whether to recompute if the output file already exists')
    parser.add_argument('--frame_size', '-fs', type=int, default=512)
    parser.add_argument('--hop_size', '-hs', type=int, default=256)
    parser.add_argument('--number_bands', '-nb', type=int, default=48)
    parser.add_argument('--sample_rate', '-sr', type=float, default=1.6e4)
    parser.add_argument('--max_frequency', '-hf', type=float, default=8.e3,
                        help='Maximum frequqncy')
    parser.add_argument('--min_frequency', '-lf', type=float, default=0.,
                        help='Minimum frequqncy')
    parser.add_argument('--window_type', '-wt', type=str, default='hann',
                        help='Essentia Windowing type. Cheeck')
    parser.add_argument('--compression_type', '-ct', type=str, default='dB',
                        help='dB: 10log10(x). log: log(1 + 10000 * x)')

    melbands_extractor(parser.parse_args())
