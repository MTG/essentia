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
from essentia import array as esarray

from librosa.effects import trim

sys.path.insert(0, './')
from qa_testevents import QaTestEvents
from qa_test import *


order = 3
frame_size = 512
sample_rate = 44100.
hop_size = 256
kernel_size = 7
times_thld = 8
energy_thld = 0.001
sub_frame = 32


class DevWrap(QaWrapper):
    """
    Development Solution.
    """
    errors = []
    errors_filt = []
    samples_peaking_frame = []
    frame_idx = []
    frames = []
    power = []

    def compute(self, *args):
        x = args[1]
        LPC = es.LPC(order=order, type='regular')
        W = es.Windowing(size=frame_size, zeroPhase=False, type='triangular')
        predicted = np.zeros(hop_size)
        y = []
        self.frames = []
        self.errors = []
        self.errors_filt = []
        self.samples_peaking_frame = []
        self.frame_idx = []
        self.power = []
        frame_counter = 0

        for frame in es.FrameGenerator(x, frameSize=frame_size,
                                       hopSize=hop_size,
                                       startFromZero=True):
            self.power.append(es.essentia.instantPower(frame))
            self.frames.append(frame)
            frame_un = np.array(frame[hop_size // 2: hop_size * 3 // 2])
            frame = W(frame)
            norm = np.max(np.abs(frame))
            if not norm:
                continue
            frame /= norm

            lpc_f, _ = LPC(esarray(frame))

            lpc_f1 = lpc_f[1:][::-1]

            for idx, i in enumerate(range(hop_size // 2, hop_size * 3 // 2)):
                predicted[idx] = - np.sum(np.multiply(frame[i - order:i], lpc_f1))

            error = np.abs(frame[hop_size // 2: hop_size * 3 // 2] - predicted)

            threshold1 = times_thld * np.std(error)

            med_filter = medfilt(error, kernel_size=kernel_size)
            filtered = np.abs(med_filter - error)

            mask = []
            for i in range(0, len(error), sub_frame):
                r = es.essentia.instantPower(frame_un[i:i + sub_frame]) > energy_thld
                mask += [r] * sub_frame
            mask = mask[:len(error)]
            mask = np.array([mask]).astype(float)[0]

            if sum(mask) == 0:
                threshold2 = 1000  # just skip silent frames
            else:
                threshold2 = times_thld * (np.std(error[mask.astype(bool)]) +
                                           np.median(error[mask.astype(bool)]))

            threshold = np.max([threshold1, threshold2])

            samples_peaking = np.sum(filtered >= threshold)
            if samples_peaking >= 1:
                y.append(frame_counter * hop_size / 44100.)
                self.frame_idx.append(frame_counter)

            self.frames.append(frame)
            self.errors.append(error)
            self.errors_filt.append(filtered)
            self.samples_peaking_frame.append(samples_peaking)

            frame_counter += 1

        return np.array(y)


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.DiscontinuityDetector(frameSize=frame_size, hopSize=hop_size)

    def compute(self, *args):
        x = args[1]
        y = []
        self.algo.reset()
        for idx, frame in enumerate(es.FrameGenerator(x, frameSize=frame_size,
                                    hopSize=hop_size,
                                    startFromZero=True)):
            locs, amps = self.algo(frame)
            for l in locs:
                y.append((l + hop_size * idx) / sample_rate)
        return esarr(y)


if __name__ == '__main__':
    folder = 'discontinuitydetector'

    # Instantiating wrappers
    wrappers = [
        DevWrap('events'),
        EssentiaWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    data_dir = '../../QA-audio/Discontinuities/prominent_jumps'

    # Add the testing files
    qa.load_audio(filename=data_dir)  # Works for a single
    qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times
    qa.compute_all(output_file='{}/compute.log'.format(folder))

    qa.score_all()

    precision = []
    recall = []
    f_measure = []
    for i in qa.scores.values():
        precision.append(i['Precision'])
        recall.append(i['Recall'])
        f_measure.append(i['F-measure'])

    print('Mean Precision: {}'.format(np.mean(precision)))
    print('Mean Recall: {}'.format(np.mean(recall)))
    print('Mean F-measure: {}'.format(np.mean(f_measure)))
