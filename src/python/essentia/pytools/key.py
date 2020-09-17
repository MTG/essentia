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

import argparse
import essentia.standard as es
import essentia.streaming as estr
import essentia


def key(audio, sampleRate=44100., frameSize=4096, hopSize=4096, windowType='hann', 
        minFrequency=25.0, maxFrequency=3500.0, spectralPeaksThreshold=0.0001, 
        maximumSpectralPeaks=60, hpcpSize=12, weightType='cosine', tuningFrequency=440.0,
        pcpThreshold=0.2, averageDetuningCorrection=True, spectralWhitening=True, profileType='bgate'):
    """ Reimplementation of the KeyExtractor algorithm in Python.

    Parameters:
        averageDetuningCorrection:
        bool ∈ {true,false} (default = true)
        shifts a pcp to the nearest tempered bin

        frameSize:
        integer ∈ (0,inf) (default = 4096)
        the framesize for computing tonal features

        hopSize:
        integer ∈ (0,inf) (default = 4096)
        the hopsize for computing tonal features

        hpcpSize:
        integer ∈ [12,inf) (default = 12)
        the size of the output HPCP (must be a positive nonzero multiple of 12)

        maxFrequency:
        real ∈ (0,inf) (default = 3500)
        max frequency to apply whitening to [Hz]

        maximumSpectralPeaks:
        integer ∈ (0,inf) (default = 60)
        the maximum number of spectral peaks

        minFrequency:
        real ∈ (0,inf) (default = 25)
        min frequency to apply whitening to [Hz]

        pcpThreshold:
        real ∈ [0,1] (default = 0.20000000298)
        pcp bins below this value are set to 0

        profileType:
        string ∈ {diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,faraldo,pentatonic,edmm,edma,bgate,braw} (default = "bgate")
        the type of polyphic profile to use for correlation calculation

        sampleRate:
        real ∈ (0,inf) (default = 44100)
        the sampling rate of the audio signal [Hz]

        spectralPeaksThreshold:
        real ∈ (0,inf) (default = 9.99999974738e-05)
        the threshold for the spectral peaks

        spectralWhitening:
        bool ∈ {true,false} (default = true)
        apply spectral whitening

        tuningFrequency:
        real ∈ (0,inf) (default = 440)
        the tuning frequency of the input signal

        weightType:
        string ∈ {none,cosine,squaredCosine} (default = "cosine")
        type of weighting function for determining frequency contribution

        windowType:
        string ∈ {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92} (default = "hann")
        the window type, which can be 'hamming', 'hann', 'triangular', 'square' or
        'blackmanharrisXX'



    Returns:
        TODO
        [string] key
        [string] scale
        [real] strength
    """

    loader = estr.VectorInput(audio)
    fc = estr.FrameCutter(frameSize=frameSize, hopSize=hopSize)
    w = estr.Windowing(size=frameSize, type=windowType)
    spectrum = estr.Spectrum()
    peaks = estr.SpectralPeaks(orderBy='magnitude',
                                magnitudeThreshold=spectralPeaksThreshold,
                                minFrequency=minFrequency,
                                maxFrequency=maxFrequency,
                                maxPeaks=maximumSpectralPeaks,
                                sampleRate=sampleRate)
    whitening = estr.SpectralWhitening(maxFrequency=maxFrequency,
                                       sampleRate=sampleRate)
    hpcp = estr.HPCP(bandPreset=False,
                     harmonics=4,
                     maxFrequency=maxFrequency,
                     minFrequency=minFrequency,
                     nonLinear=False,
                     normalized='none',
                     referenceFrequency=tuningFrequency,
                     sampleRate=sampleRate,
                     size=hpcpSize,
                     weightType=weightType,
                     windowSize=1.0,
                     maxShifted=False)
    key = estr.Key(usePolyphony=False,
                   useThreeChords=False,
                   numHarmonics=4,
                   slope=0.6,
                   profileType=profileType,
                   pcpSize=hpcpSize,
                   pcpThreshold=pcpThreshold,
                   averageDetuningCorrection=averageDetuningCorrection)

    p = essentia.Pool()

    loader.data >> fc.signal
    fc.frame >> w.frame >> spectrum.frame
    spectrum.spectrum >> peaks.spectrum
    
    if spectralWhitening:
        peaks.magnitudes >> whitening.magnitudes
        peaks.frequencies >> whitening.frequencies
        spectrum.spectrum >> whitening.spectrum
        whitening.magnitudes >> hpcp.magnitudes
    else:
        peaks.magnitudes >> hpcp.magnitudes

    peaks.frequencies >> hpcp.frequencies

    hpcp.hpcp >> key.pcp
    key.key >> (p, 'key')
    key.scale >> (p, 'scale')
    key.strength >> (p, 'strength')

    essentia.run(loader)
    return p['key'], p['scale'], p['strength']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate key of a given audio file.')

    parser.add_argument('filename',
                        help='the name of the input audio file')

    audio = es.MonoLoader(filename=parser.parse_args().filename)()


    print("pytools.key:", key(audio))
    print("KeyExtractor:", es.KeyExtractor()(audio))
