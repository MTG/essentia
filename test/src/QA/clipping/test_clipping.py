#!/usr/bin/env python

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


import sys

import numpy as np
from scipy.signal import medfilt

import essentia.standard as es
from essentia import array as esarr

sys.path.insert(0, './')
from qa_test import *
from qa_testevents import QaTestEvents


class DevWrap(QaWrapper):
    """
    Development Solution.
    """

    # parameters
    _sampleRate = 44100.
    _frameSize = 512
    _hopSize = 256
    _minimumDuration = 0  # ms
    _minimumDuration /= 1000.
    _energyThreshold = db2amp(-.001)
    _differentialThreshold = .1

    # inner variables
    _idx = 0
    _previousRegion = None

    def compute(self, *args):
        x = args[1]
        y = []
        self._idx = 0
        for frame in es.FrameGenerator(x, frameSize=self._frameSize,
                                       hopSize=self._hopSize,
                                       startFromZero=True):
            frame = np.abs(frame)
            starts = []
            ends = []

            s = int(self._frameSize / 2 - self._hopSize / 2) - 1  # consider non overlapping case
            e = int(self._frameSize / 2 + self._hopSize / 2)

            delta = np.diff(frame)
            delta = np.insert(delta, 0, 0)
            energyMask = np.array([x > self._energyThreshold for x in frame])[s:e].astype(int)
            deltaMask = np.array([np.abs(x) <= self._differentialThreshold for x in delta])[s:e].astype(int)

            combinedMask = energyMask * deltaMask

            flanks = np.diff(combinedMask)

            uFlanks = [idx for idx, x in enumerate(flanks) if x == 1]
            dFlanks = [idx for idx, x in enumerate(flanks) if x == -1]

            uFlanksValues = []
            uFlanksPValues = []
            for uFlank in uFlanks:
                uFlanksValues.append(frame[uFlank + s])
                uFlanksPValues.append(frame[uFlank + s - 1])

            dFlanksValues = []
            dFlanksPValues = []
            for dFlank in dFlanks:
                dFlanksValues.append(frame[dFlank + s])
                dFlanksPValues.append(frame[dFlank + s + 1])

            if self._previousRegion and dFlanks:
                start = self._previousRegion
                end = (self._idx * self._hopSize + dFlanks[0] + s) / self._sampleRate
                duration = start - end

                if duration > self._minimumDuration:
                    starts.append(start)
                    ends.append(end)

                self._previousRegion = None
                del dFlanks[0]

                del dFlanksValues[0]
                del dFlanksPValues[0]

            if len(dFlanks) is not len(uFlanks):
                self._previousRegion = (self._idx * self._hopSize + uFlanks[-1] + s) / self._sampleRate
                del uFlanks[-1]

            if len(dFlanks) is not len(uFlanks):
                raise EssentiaException(
                    "Ath this point uFlanks ({}) and dFlanks ({}) are expected to have the same length!".format(len(dFlanks),
                                                                                                            len(uFlanks)))

            for idx in range(len(uFlanks)):
                start = float(self._idx * self._hopSize + uFlanks[idx] + s) / self._sampleRate
                end = float(self._idx * self._hopSize + dFlanks[idx] + s) / self._sampleRate
                duration = end - start
                if duration > self._minimumDuration:
                    xs = [uFlanks[idx] - 1, uFlanks[idx], dFlanks[idx], dFlanks[idx] + 1]
                    ys = [uFlanksPValues[idx], uFlanksValues[idx], dFlanksValues[idx], dFlanksPValues[idx]]

                    coefs = np.polyfit(xs, ys, 2)

                    starts.append(start)
                    ends.append(end)

                    estx, esty = self.maxParable(coefs)

                    if esty > 1.0:
                        starts.append(start)

                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        plt.axvline(s, color='r')
                        plt.axvline(e, color='r')
                        plt.plot(frame, label='audio')

                        xs = s + uFlanks[idx]
                        plt.axvline(xs, color='g', alpha=.2)
                        xs = s + dFlanks[idx]
                        plt.axvline(xs, color='g', alpha=.2)

                        xs = [uFlanks[idx] - 1, uFlanks[0], dFlanks[0], dFlanks[0] + 1]
                        xs2 = np.array(xs) + s

                        plt.plot(xs2, ys, 'ro', label='points used for the parable estimation')
                        plt.plot(estx + s, esty, 'yo', label='estimated audio peak')

                        x3 = np.linspace(xs[0], xs[-1], 10)
                        y3 = [self.parEval(xx, coefs) for xx in x3]
                        plt.plot(x3 + s, y3, label='estimated parable', alpha=.2)

                        plt.axhline(1.0, color='g', ls='--', alpha=.2, label='maximun dynamic range')

                        plt.title('Parabolic Regression for Clipping Detection')
                        plt.xlim(xs2[0]-15, xs2[0] + 15)
                        plt.legend()
                        plot_name = 'clipping_plots/{}_{}'.format(self._idx, uFlanks[idx])
                        plt.savefig(plot_name)
                        plt.clf()

            for start in starts:
                y.append(start)
            self._idx += 1

        return esarr(y)

    def parEval(self, x, coefs):
        return coefs[0] * x ** 2 + coefs[1] * x + coefs[2]

    def maxParable(self, coefs):
        xm = -coefs[1] / (2 * coefs[0])
        return xm, self.parEval(xm, coefs)


class DevWrap2(QaWrapper):
    """
    Development Solution.
    """

    # parameters
    _sampleRate = 44100.
    _frameSize = 512
    _hopSize = 256
    _minimumDuration = 0  # ms
    _minimumDuration /= 1000.
    _energyThreshold = db2amp(-.001)
    _differentialThreshold = .001

    # inner variables
    _idx = 0
    _previousRegion = None

    def compute(self, *args):
        x = args[1]
        y = []
        self._idx = 0
        for frame in es.FrameGenerator(x, frameSize=self._frameSize, 
                                       hopSize=self._hopSize,
                                       startFromZero=True):
            frame = np.abs(frame)
            starts = []
            ends = []

            s = int(self._frameSize / 2 - self._hopSize / 2) - 1  # consider non overlapping case
            e = int(self._frameSize / 2 + self._hopSize / 2)

            for idx in range(s, e):
                if frame[idx] >= self._energyThreshold:
                    continue

        return esarr(y)


class TruePeakDetector(QaWrapper):
    # Frame-wise implementation following ITU-R BS.1770-2

    # parameters
    _sampleRate = 44100.
    _frameSize = 512
    _hopSize = 256
    _oversample = 4
    _sampleRateOver = _oversample * _sampleRate
    _quality = 1
    _BlockDC = False
    _emphatise = False

    # inner variables
    _idx = 0
    _clippingThreshold = 0.9999695

    def compute(self, *args):
        from math import pi

        x = args[1]
        for frame in es.FrameGenerator(x, frameSize=self._frameSize,
                                       hopSize=self._hopSize, startFromZero=True):
            y = []
            s = int(self._frameSize / 2 - self._hopSize / 2) - 1  # consider non overlapping case
            e = int(self._frameSize / 2 + self._hopSize / 2)

            # Stage 1: Attenuation. Is not required because we are using float point.

            # Stage 2: Resample
            yResample = es.Resample(inputSampleRate=self._sampleRate,
                                    outputSampleRate=self._sampleRateOver,
                                    quality=self._quality)(frame)

            # Stage 3: Emphasis
            if self._emphatise:
                fPole = 20e3  # Hz
                fZero = 14.1e3

                rPole = fPole / self._sampleRateOver
                rZero = fZero / self._sampleRateOver

                yEmphasis = es.IIR(denominator=esarr([1., rPole]),
                                   numerator=esarr([1., -rZero]))(yResample)
            else:
                yEmphasis = yResample

            # Stage 4 Absolute
            yMaxArray = np.abs(yEmphasis)

            # Stage 5 optional DC Block
            if self._BlockDC:
                yDCBlocked = es.DCRemoval(sampleRate=self._sampleRate,
                                          cutoffFrequency=1.)(yEmphasis)

                yAbsoluteDCBlocked = np.abs(yDCBlocked)

                yMaxArray = np.maximum(yMaxArray, yAbsoluteDCBlocked)

            y = [((i + self._idx * self._hopSize) / float(self._sampleRateOver), yMax)
                 for i, yMax in enumerate(yMaxArray) if yMax > self._clippingThreshold]

            self._idx += 1

        return esarr(y)


if __name__ == '__main__':
    folder = 'clipping'

    # Instantiating wrappers
    wrappers = [
        TruePeakDetector('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    data_dir = '../../audio/recorded/distorted.wav'

    qa.load_audio(filename=data_dir)  # Works for a single
    # qa.load_solution(data_dir, ground_true=True)

    # Compute all the estimations, get the scores and compare the computation times
    qa.compute_all(output_file='{}/compute.log'.format(folder))

    # Optional plotting
    # qa.plot_all('clipping_plots/')
