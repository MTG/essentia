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
