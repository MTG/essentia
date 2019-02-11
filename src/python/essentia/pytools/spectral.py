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

import numpy as np
import essentia.standard as es
from essentia import Pool, array


def nsgcqgram(audio, frameSize=8192, transitionSize=1024, minFrequency=65.41,
              maxFrequency=6000, binsPerOctave=48,
              sampleRate=44100, rasterize='full',
              phaseMode='global', gamma=0,
              normalize='none', window='hannnsgcq'):
    """Frame-wise invertible Constant-Q analysis.
    This code replicates the Sli-CQ algorithm from [1]. A Tukey window
    is used to perform a zero-phased, zero-padded, half-overlapped
    frame-wise analysis with the `NSGConstantQ` algorithm.

    References:
      [1] Velasco, G. A., Holighaus, N., Dörfler, M., & Grill, T. (2011).
        "Constructing an invertible constant-Q transform with non-stationary
        Gabor frames". Proceedings of DAFX11, Paris, 93-99.

    Args:
        audio (vector): If it is empty, an exception is raised.
    Returns:
        (list of 2D complex arrays): Time/frequency complex matrices representing the NSGCQ `constantq` coefficients for each `frameSize // 2` samples jump.
        (list of complex vectors): Complex vectors representing the NSGCQ `constantqdc` coefficients for each `frameSize // 2` samples jump.
        (list of complex vectors): Complex vectors representing the NSGCQ `constantqnf` coefficients for each `frameSize // 2` samples jump.
    """

    hopSize = frameSize // 2
    halfTransitionSize = transitionSize // 2

    NSGCQ = es.NSGConstantQ(inputSize=frameSize, minFrequency=minFrequency,
                            maxFrequency=maxFrequency, binsPerOctave=binsPerOctave,
                            sampleRate=sampleRate, rasterize=rasterize,
                            phaseMode=phaseMode, gamma=gamma,
                            normalize=normalize, window=window,
                            minimumWindow=8)

    w = es.Windowing(type='hannnsgcq', normalized=False, zeroPhase=False)(
        np.ones(transitionSize * 2).astype('float32'))
    window = np.hstack(
        [w[-transitionSize:], np.ones(hopSize - transitionSize), w[:transitionSize]])

    # Indexes for the windowing process.
    evenWin = np.hstack([np.arange(frameSize - halfTransitionSize,
                                   frameSize), np.arange(hopSize + halfTransitionSize)])
    oddWin = np.hstack([np.arange(hopSize - halfTransitionSize,
                                  frameSize), np.arange(halfTransitionSize)])
    h0 = np.arange(-halfTransitionSize, hopSize + halfTransitionSize)

    # Zero-pad the signal for an integer number of frames.
    zeroPad = (frameSize - window.size) // 2
    frameNum = int(np.ceil(audio.size / hopSize))

    x = np.hstack([audio, np.zeros(hopSize * frameNum -
                                   audio.size + hopSize)]).astype('float32')
    frames = np.zeros([frameNum, frameSize], dtype='float32')

    # Mirror-pad for the first frame.
    frames[0][evenWin] = np.hstack(
        [-x[halfTransitionSize - 1::-1], x[:hopSize + halfTransitionSize]]) * window

    # Slice audio.
    for kk in range(2, frameNum, 2):
        frames[kk][evenWin] = x[(kk - 1) * hopSize + h0] * window

    for kk in range(1, frameNum, 2):
        frames[kk][oddWin] = x[(kk - 1) * hopSize + h0] * window

    # Compute the transform.
    cqShift, dcShift, nfShift = [], [], []
    for i in range(frameNum):
        cqFrame, dcFrame, nfFrame = NSGCQ(frames[i])
        cqShift.append(cqFrame)
        dcShift.append(dcFrame)
        nfShift.append(nfFrame)

    cqSize = cqFrame.shape[1]
    dcSize = dcFrame.size
    nfSize = nfFrame.size

    # Center the frames for a better display.
    cqShiftEven = np.hstack(
        [np.arange(cqSize // 4, cqSize), np.arange(0, cqSize // 4)])
    dcShiftEven = np.hstack(
        [np.arange(dcSize // 4, dcSize), np.arange(0, dcSize // 4)])
    nfShiftEven = np.hstack(
        [np.arange(nfSize // 4, nfSize), np.arange(0, nfSize // 4)])

    cqShiftOdd = np.hstack(
        [np.arange(3 * cqSize // 4, cqSize), np.arange(0, 3 * cqSize // 4)])
    dcShiftOdd = np.hstack(
        [np.arange(3 * dcSize // 4, dcSize), np.arange(0, 3 * dcSize // 4)])
    nfShiftOdd = np.hstack(
        [np.arange(3 * nfSize // 4, nfSize), np.arange(0, 3 * nfSize // 4)])

    cq = [cqShift[i][:, cqShiftEven] if i % 2 else cqShift[i][:, cqShiftOdd]
          for i in range(len(cqShift))]
    dc = [dcShift[i][dcShiftEven] if i % 2 else dcShift[i][dcShiftOdd]
          for i in range(len(dcShift))]
    nf = [nfShift[i][nfShiftEven] if i % 2 else nfShift[i][nfShiftOdd]
          for i in range(len(nfShift))]

    return cq, dc, nf


def __inverseTukeyWindow__(x):
    return (1 + np.cos(np.pi * x)) / (1 + np.cos(np.pi * x) ** 2)


def nsgicqgram(cq, dc, nf, frameSize=8192, transitionSize=1024, minFrequency=65.41,
               maxFrequency=6000, binsPerOctave=48,
               sampleRate=44100, rasterize='full',
               phaseMode='global', gamma=0,
               normalize='none', window='hannnsgcq'):
    """Frame-wise invertible Constant-Q synthesis.
    This code replicates the Sli-CQ algorithm from [1]. An inverse Tukey window
    is used to resynthetise the original audio signal from the `nsgcqgram`
    representation using the `NSGIConstantQ` algorithm.

    References:
      [1] Velasco, G. A., Holighaus, N., Dörfler, M., & Grill, T. (2011).
        "Constructing an invertible constant-Q transform with non-stationary
        Gabor frames". Proceedings of DAFX11, Paris, 93-99.

    Args:
        (list of 2D complex arrays): Time / frequency complex matrices representing the NSGCQ `constantq` coefficients for each `frameSize // 2` samples jump.
        (list of complex vectors): Complex vectors representing the NSGCQ `constantqdc` coefficients for each `frameSize // 2` samples jump.
        (list of complex vectors): Complex vectors representing the NSGCQ `constantqnf` coefficients for each `frameSize // 2` samples jump.
    Returns:
        audio (vector): The synthetized audio.
    """

    hopSize = frameSize // 2
    halfTransitionSize = transitionSize // 2

    cqSize = cq[0].shape[1]
    dcSize = len(dc[0])
    nfSize = len(nf[0])

    NSGICQS = es.NSGIConstantQ(inputSize=frameSize, minFrequency=minFrequency,
                               maxFrequency=maxFrequency, binsPerOctave=binsPerOctave,
                               sampleRate=sampleRate, rasterize=rasterize,
                               phaseMode=phaseMode, gamma=gamma,
                               normalize=normalize, window=window,
                               minimumWindow=8)

    # Tukey inverse window.
    window = np.zeros(frameSize)
    window[np.arange((hopSize + transitionSize) // 2,
                     (3 * hopSize - transitionSize) // 2)
           ] = np.ones(hopSize - transitionSize)

    window[np.hstack([np.arange((hopSize - transitionSize) // 2,
                                (hopSize + transitionSize) // 2),
                      np.arange((3 * hopSize - transitionSize) // 2,
                                (3 * hopSize + transitionSize) // 2)])
           ] = __inverseTukeyWindow__(np.arange(-transitionSize,
                                                transitionSize) / transitionSize)

    # Undo the frame centering.
    cqShiftEven = np.hstack(
        [np.arange(3 * cqSize // 4, cqSize), np.arange(0, 3 * cqSize // 4)])
    dcShiftEven = np.hstack(
        [np.arange(3 * dcSize // 4, dcSize), np.arange(0, 3 * dcSize // 4)])
    nfShiftEven = np.hstack(
        [np.arange(3 * nfSize // 4, nfSize), np.arange(0, 3 * nfSize // 4)])

    cqShiftOdd = np.hstack(
        [np.arange(cqSize // 4, cqSize), np.arange(0, cqSize // 4)])
    dcShiftOdd = np.hstack(
        [np.arange(dcSize // 4, dcSize), np.arange(0, dcSize // 4)])
    nfShiftOdd = np.hstack(
        [np.arange(nfSize // 4, nfSize), np.arange(0, nfSize // 4)])

    cqShift = [cq[i][:, cqShiftEven] if i % 2 else cq[i][:, cqShiftOdd]
          for i in range(len(cq))]
    dcShift = [dc[i][dcShiftEven] if i % 2 else dc[i][dcShiftOdd]
          for i in range(len(dc))]
    nfShift = [nf[i][nfShiftEven] if i % 2 else nf[i][nfShiftOdd]
          for i in range(len(nf))]


    # Loop to store the audio frames.
    frames = [NSGICQS(cqShift[i], dcShift[i], nfShift[i])
              for i in range(len(cq))]


    # Overlap-add.
    audio = np.zeros((len(frames) + 1) * hopSize)

    for kk in range(len(frames)):
        audio[np.arange(int(np.ceil((kk - .5) * hopSize)),
                        int(np.ceil((kk + 1.5) * hopSize)))
              ] += np.roll(frames[kk], int(np.floor(((-1) ** kk) * hopSize / 2))) * window

    return audio[hopSize:]


def nsgcq_overlap_add(cq):
    """Frame-wise invertible Constant-Q synthesis.
    This function performs the overlap-add process of the CQ frames obtained by nsgcq_gram.
    The output of this algorithm may be used for visualization purposes.
    Note: It is not possible to perform a perfect reconstruction from the overlapped version of the CQ data. 

    Args:
        (list of 2D complex arrays): Time / frequency complex matrices representing the NSGCQ `constantq` coefficients for each `frameSize // 2` samples jump.
    Returns:
        (2D complex array): The overlapped version of the Constant-Q. 
    """

    frameNum = len(cq)
    cqChannels = cq[0].shape[0]
    cqSize = cq[0].shape[1]

    hopSize = cqSize // 2
    timeStamps = (frameNum + 1) * hopSize

    index = np.arange(cqSize)

    cqOverlap = np.zeros(
        [cqChannels, timeStamps], dtype='complex')


    # Overlap-add.
    for jj in range(frameNum):
        cqOverlap[:, jj * hopSize + index] += cq[jj]

    return cqOverlap[:, hopSize:]


def hpcp(audio, sampleRate=44100, frameSize=4096, hopSize=2048, windowType='blackmanharris62', 
        harmonicsPerPeak=8, magnitudeThreshold=1e-05, maxPeaks=1000, whitening=True, numBins=12, 
        referenceFrequency=440, minFrequency=40,maxFrequency=5000, nonLinear=False):
    """
    Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
    the default parameters as mentioned in [1]. 

    Please refer to the following paper for detailed explanantion of the algorithm.
    [1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing.

    For full list of parameters of essentia standard mode HPCP please refer to 
    http://essentia.upf.edu/documentation/reference/std_HPCP.html

    Inputs
        audio (2d vector): audio signal

    Parameters:
        harmonicsPerPeak : (integer ∈ [0, ∞), default = 0) :
        number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution
        maxFrequency : (real ∈ (0, ∞), default = 5000) :
        the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz)

        minFrequency : (real ∈ (0, ∞), default = 40) :
        the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz)

        nonLinear : (bool ∈ {true, false}, default = false) :
        apply non-linear post-processing to the output (use with normalized='unitMax'). Boosts values close to 1, decreases values close to 0.
        normalized (string ∈ {none, unitSum, unitMax}, default = unitMax) :
        whether to normalize the HPCP vector

        referenceFrequency : (real ∈ (0, ∞), default = 440) :
        the reference frequency for semitone index calculation, corresponding to A3 [Hz]

        sampleRate : (real ∈ (0, ∞), default = 44100) :
        the sampling rate of the audio signal [Hz]

        numBins : (integer ∈ [12, ∞), default = 12) :
        the size of the output HPCP (must be a positive nonzero multiple of 12)
        whitening : (boolean (True, False), default = False)
        Optional step of computing spectral whitening to the output from speakPeak magnitudes

    Returns: hpcp (2D vector) 

    """
    audio = array(audio)
    frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)
    window = estd.Windowing(type=windowType)
    spectrum = estd.Spectrum()
    # Refer http://essentia.upf.edu/documentation/reference/std_SpectralPeaks.html
    spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=magnitudeThreshold,
                                        maxFrequency=maxFrequency,
                                        minFrequency=minFrequency,
                                        maxPeaks=maxPeaks,
                                        orderBy="frequency",
                                        sampleRate=sampleRate)
    # http://essentia.upf.edu/documentation/reference/std_SpectralWhitening.html
    spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency,
                                                    sampleRate=sampleRate)
    # http://essentia.upf.edu/documentation/reference/std_HPCP.html
    hpcp = estd.HPCP(sampleRate=sampleRate,
                    maxFrequency=maxFrequency,
                    minFrequency=minFrequency,
                    referenceFrequency=referenceFrequency,
                    nonLinear=nonLinear,
                    harmonics=harmonicsPerPeak,
                    size=numBins)
    pool = Pool()
    #compute hpcp for each frame and add the results to the pool
    for frame in frameGenerator:
        spectrum_mag = spectrum(window(frame))
        frequencies, magnitudes = spectralPeaks(spectrum_mag)
        if whitening:
            w_magnitudes = spectralWhitening(spectrum_mag,
                                            frequencies,
                                            magnitudes)
            hpcp_vector = hpcp(frequencies, w_magnitudes)
        else:
            hpcp_vector = hpcp(frequencies, magnitudes)
        pool.add('tonal.hpcp',hpcp_vector)

        return pool['tonal.hpcp']

