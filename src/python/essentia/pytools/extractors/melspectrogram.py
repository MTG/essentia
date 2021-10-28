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

import argparse
import os

import numpy as np

from essentia import Pool
from essentia import run

from essentia.streaming import (MonoLoader, FrameCutter, Windowing, Spectrum,
                                MelBands, UnaryOperator)

ZERO_PADDING = 0
WINDOW_TYPE = 'hann'
FRAME_SIZE = 1024
HOP_SIZE = 512
NUMBER_BANDS = 24
SAMPLE_RATE = 44100.
LOW_FREQUENCY_BOUND = 0.
HIGH_FREQUENCY_BOUND = 22050.
WARPING_FORMULA = 'htkMel'
WEIGHTING = 'warping'
NORMALIZE = 'unit_sum'
BANDS_TYPE = 'power'
COMPRESSION_TYPE = 'shift_scale_log'


def melspectrogram(filename, npy_file=None, force=False, verbose=False, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE,
    hop_size=HOP_SIZE, window_type=WINDOW_TYPE, zero_padding=ZERO_PADDING, low_frequency_bound=LOW_FREQUENCY_BOUND,
    high_frequency_bound=HIGH_FREQUENCY_BOUND, number_bands=NUMBER_BANDS, warping_formula=WARPING_FORMULA,
    weighting=WEIGHTING, normalize=NORMALIZE, bands_type=BANDS_TYPE, compression_type=COMPRESSION_TYPE):
    """Computes the mel spectrogram given the audio filename.
    When the parameter `npy_file` is specified, the data is saved to disk as a numpy array (.npy).
    Use the parameter `force` to overwrite the numpy array in case it already exists.
    The rest of parameters are directly mapped to Essentia algorithms as explained below.

    Note: this functionality is also available as a command line script.

    Parameters:
        sample_rate:
        real ∈ (0,inf) (default = 44100)
        the desired output sampling rate [Hz]

        frame_size:
        integer ∈ [1,inf) (default = 1024)
        the output frame size

        hop_size:
        integer ∈ [1,inf) (default = 512)
        the hop size between frames

        window_type:
        string ∈ {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92} (default = "hann")
        the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'

        zero_padding:
        integer ∈ [0,inf) (default = 0)
        the size of the zero-padding

        low_frequency_bound:
        real ∈ [0,inf) (default = 0)
        a lower-bound limit for the frequencies to be included in the bands

        high_frequency_bound:
        real ∈ [0,inf) (default = 22050)
        an upper-bound limit for the frequencies to be included in the bands

        number_bands:
        integer ∈ (1,inf) (default = 24)
        the number of output bands

        warping_formula:
        string ∈ {slaneyMel,htkMel} (default = "htkMel")
        The scale implementation type: 'htkMel' scale from the HTK toolkit [2, 3]
        (default) or 'slaneyMel' scale from the Auditory toolbox [4]

        weighting:
        string ∈ {warping,linear} (default = "warping")
        type of weighting function for determining triangle area

        normalize:
        string ∈ {unit_sum,unit_tri,unit_max} (default = "unit_sum")
        spectrum bin weights to use for each mel band: 'unit_max' to make each mel
        band vertex equal to 1, 'unit_sum' to make each mel band area equal to 1
        summing the actual weights of spectrum bins, 'unit_area' to make each
        triangle mel band area equal to 1 normalizing the weights of each triangle
        by its bandwidth

        bands_type:
        string ∈ {magnitude,power} (default = "power")
        'power' to output squared units, 'magnitude' to keep it as the input

        compression_type:
        string ∈ {dB,shift_scale_log,none} (default = "shift_scale_log")
        the compression type to use.
        'shift_scale_log' is log10(10000 * x + 1)
        'dB' is 10 * log10(x)

    Returns:
        (2D array): The mel-spectrogram.
    """

    padded_size = frame_size + zero_padding
    spectrum_size = (padded_size) // 2 + 1


    # In case we want to save the melbands to a file
    # check if the file already exists
    if npy_file:
        if not npy_file.endswith('.npy'):
            npy_file += '.npy'

        if not force:
            if os.path.exists(npy_file):
                if verbose:
                    print('Skipping "{}"'.format(npy_file))
                return

    pool = Pool()

    loader = MonoLoader(filename=filename,
                        sampleRate=sample_rate)
    frameCutter = FrameCutter(frameSize=frame_size,
                              hopSize=hop_size)
    w = Windowing(zeroPadding=zero_padding,
                  type=window_type,
                  normalized=False)  # None of the mel bands extraction methods
                                     # we have seen requires window-level normalization.
    spec = Spectrum(size=padded_size)
    mels = MelBands(inputSize=spectrum_size,
                    numberBands=number_bands,
                    sampleRate=sample_rate,
                    lowFrequencyBound=low_frequency_bound,
                    highFrequencyBound=high_frequency_bound,
                    warpingFormula=warping_formula,
                    weighting=weighting,
                    normalize=normalize,
                    type=bands_type,
                    log=False)  # Do not compute any compression here.
                                # Use the `UnaryOperator`s methods before
                                # in case a new compression type is required.

    if compression_type.lower() == 'db':
        shift = UnaryOperator(type='identity')
        compressor = UnaryOperator(type='lin2db')

    elif compression_type.lower() == 'shift_scale_log':
        shift = UnaryOperator(type='identity', scale=1e4, shift=1)
        compressor = UnaryOperator(type='log10')

    elif compression_type.lower() == 'none':
        shift = UnaryOperator(type='identity')
        compressor = UnaryOperator(type='identity')

    loader.audio >> frameCutter.signal
    frameCutter.frame >> w.frame >> spec.frame
    spec.spectrum >> mels.spectrum
    mels.bands >> shift.array >> compressor.array >> (pool, 'mel_bands')

    run(loader)

    mel_bands = np.array(pool['mel_bands'])

    if npy_file:
        np.save(npy_file, mel_bands)
        
    if verbose:
        print('Done for "{}"'.format(npy_file))

    return mel_bands


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes the mel spectrogram of a given audio file.')

    parser.add_argument('filename',
                        help='the name of the file from which to read')
    parser.add_argument('npy_file', type=str,
                        help='the name of the output file')
    parser.add_argument('--force', '-f', action='store_true',
                        help='whether to recompute if the output file already exists')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='whether to print out status to the standard output')
    parser.add_argument('--sample-rate', '-sr', type=float, default=SAMPLE_RATE,
                        help='the sample rate')
    parser.add_argument('--frame-size', '-fs', type=int, default=FRAME_SIZE,
                        help='the output frame size')
    parser.add_argument('--hop-size', '-hs', type=int, default=HOP_SIZE,
                        help='the hop size between frames')
    parser.add_argument('--window-type', '-wt', type=str, default=WINDOW_TYPE,
                        help='window type', choices=('hamming', 'hann', 'hannnsgcq', 'triangular', 'square', 'blackmanharris62',
                                                     'blackmanharris70', 'blackmanharris74', 'blackmanharris92'))
    parser.add_argument('--zero-padding', '-zp', type=int, default=ZERO_PADDING,
                        help='the size of the zero-padding')
    parser.add_argument('--low-frequency-bound', '-lf', type=float, default=LOW_FREQUENCY_BOUND,
                        help='a lower-bound limit for the frequencies to be included in the bands')
    parser.add_argument('--high-frequency-bound', '-hf', type=float, default=HIGH_FREQUENCY_BOUND,
                        help='an upper-bound limit for the frequencies to be included in the bands')
    parser.add_argument('--number-bands', '-nb', type=int, default=NUMBER_BANDS,
                        help='the number of output bands')
    parser.add_argument('--warping-formula', '-wf', type=str, default=WARPING_FORMULA, choices=('slaneyMel','htkMel'),
                        help='the scale implementation type: `htkMel` scale from the HTK toolkit(default) or `slaneyMel` scale from the Auditory toolbox')
    parser.add_argument('--weighting', '-we', type=str, default=WEIGHTING, choices=('warping','linear'),
                        help='type of weighting function for determining triangle area')
    parser.add_argument('--normalize', '-n', type=str, default=NORMALIZE, choices=('unit_sum', 'unit_tri', 'unit_max'),
                        help='spectrum bin weights to use for each mel band: `unit_max` to make each mel band vertex equal to 1, `unit_sum` to make each mel band area equal to 1 summing the actual weights of spectrum bins, `unit_area` to make each triangle mel band area equal to 1 normalizing the weights of each triangle by its bandwidth')
    parser.add_argument('--bands-type', '-bt', type=str, default=BANDS_TYPE, choices=('magnitude','power'),
                        help='`power` to output squared units, `magnitude` to keep it as the input')
    parser.add_argument('--compression-type', '-ct', type=str, default=COMPRESSION_TYPE, choices=('dB', 'shift_scale_log', 'none'),
                        help='dB: 10log10(x). shift_scale_log: log(1 + 10000 * x)')

    melspectrogram(**vars(parser.parse_args()))
