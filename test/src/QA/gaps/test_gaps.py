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

import essentia.standard as es
from essentia import array as esarray
from essentia import instantPower
from essentia import array as esarr

sys.path.insert(0, './')
from qa_test import *
from qa_testevents import QaTestEvents


# parameters
frame_size = 2048
hop_size = 1024

class DevWrap(QaWrapper):
    """
    Developmet Solution.
    """

    # parameters
    frame_size = frame_size
    hop_size = hop_size
    fs = 44100.

    threshold = -50
    prepower_threshold = -30

    prepower_time = .04 #s

    min_time = .01  #s
    max_time = 3.5  #s

    attackTime = .05
    releaseTime = .05

    # private variables
    _threshold = es.essentia.db2amp(threshold)
    _prepower_threshold = es.essentia.db2amp(prepower_threshold) ** 2
    _prepower_samples = int(prepower_time * fs)
    l_buffer = np.zeros(_prepower_samples)
    _gaps = []

    envelope = es.Envelope(releaseTime=releaseTime, attackTime=attackTime)
    medianFilter = es.MedianFilter()

    def compute(self, *args):
        y = []
        x = args[1]
        for frame_idx, frame in enumerate(es.FrameGenerator(x, frameSize=self.frame_size,
                                          hopSize=self.hop_size, startFromZero=True)):
            # frame = es.essentia.normalize(frame)
            # updating buffers
            for gap in self._gaps:
                if not gap['finished'] and not gap['active']:
                    last = np.min([self.frame_size, gap['take']])
                    gap['take'] -= last
                    gap['buffer'] = np.hstack([gap['buffer'], frame[:last]])
                    if gap['take'] <= 0:
                        gap['finished'] = True
            remove_idx = []
            for gap_idx, gap in enumerate(self._gaps):
                if gap['finished']:
                    remove_idx.append(gap_idx)
                    postpower = instantPower(esarr(gap['buffer']))
                    if postpower > self._prepower_threshold:
                        if self.min_time <= gap['end'] - gap['start'] <= self.max_time:
                            y.append(gap['start'])

            remove_idx.sort(reverse=True)
            for i in remove_idx:
                self._gaps.pop(i)

            x1 = self.envelope(frame)
            x2 = esarr(x1 > self._threshold)

            x3 = self.medianFilter(x2).round().astype(int)

            x3_d = np.zeros(len(x3))

            start_proc = int(self.frame_size / 2 - self.hop_size / 2)
            end_proc = int(self.frame_size / 2 + self.hop_size / 2)
            for i in range(start_proc, end_proc):

                x3_d[i] = x3[i] - x3[i-1]

            s_dx = np.argwhere(x3_d == -1)
            e_dx = np.argwhere(x3_d == 1)

            # initializing
            if s_dx.size:
                offset = frame_idx * self.hop_size
                for s in s_dx:
                    s = s[0]
                    take_from_buffer = s - self._prepower_samples
                    if take_from_buffer > 0:
                        prepower = instantPower(frame[take_from_buffer:s])
                    else:
                        prepower = instantPower(esarr(np.hstack([self.l_buffer[-np.abs(take_from_buffer):],
                                                frame[:s]])))
                    if prepower > self._prepower_threshold:
                        self._gaps.append({'start': (offset + s) / self.fs,
                                          'end': 0,
                                          'buffer': [],
                                          'take': 0,
                                          'active': True,
                                          'finished': False})

            # finishing
            if e_dx.size and self._gaps:
                offset = frame_idx * self.hop_size
                for e in e_dx:
                    e = e[0]
                    take_from_next_frame = np.max([(self._prepower_samples + e) - self.frame_size, 0])
                    for gap in self._gaps:
                        if gap['active']:
                            gap['take'] = take_from_next_frame
                            gap['end'] = (offset + e) / self.fs
                            last = np.min([self.frame_size, e + self._prepower_samples])
                            gap['buffer'] = frame[e: last]
                            gap['active'] = False
                            break

            # update buffers
            update_num = np.min([self._prepower_samples, self.hop_size])
            np.roll(self.l_buffer, -update_num)
            self.l_buffer[-update_num:] = frame[-update_num:]

        self._gaps = []
        return esarr(y)


class EssentiaWrap(QaWrapper):
    """
    Implemented Solution.
    """
    def compute(self, *args):
        y = []
        x = args[1]
        gapDetector = es.GapsDetector()

        for frame in es.FrameGenerator(x, frameSize=frame_size,
                                       hopSize=hop_size, startFromZero=True):
            starts, _ = gapDetector(frame)
            for s in starts:
                y.append(s)
        return esarr(y)


if __name__ == '__main__':

    # Instantiating wrappers
    wrappers = [
        DevWrap('events', ground_true=True),
        EssentiaWrap('events', ground_true=True),
    ]

    # Instantiating the test
    qa = QaTestEvents(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # Add the testing files
    data_dir = '../../QA-audio/Gaps/forced/'

    qa.load_audio(filename=data_dir)  # Works for a single
    qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times
    qa.compute_all()

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
