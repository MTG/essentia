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


from qa_test import *
from qa_testevents import QaTestEvents
import numpy as np
from librosa.effects import trim
import essentia.standard as es
from essentia import array as esarray


class EssentiaWrap60(QaWrapper):
    """
    Essentia Solution.
    """

    def compute(self, x):
        startstopsilence = es.StartStopSilence(threshold=-60)
        for frame in es.FrameGenerator(x, frameSize=2048, hopSize=512, startFromZero=True):
            result = startstopsilence(esarray(frame))
        solution = np.array([result[0], result[1]])
        return solution * 512 / 44100.


class EssentiaWrap50(QaWrapper):
    """
    Essentia Solution.
    """

    def compute(self, x):
        startstopsilence = es.StartStopSilence(threshold=-50)
        for frame in es.FrameGenerator(x, frameSize=2048, hopSize=512, startFromZero=True):
            result = startstopsilence(esarray(frame))
        solution = np.array([result[0], result[1]])
        return solution * 512 / 44100.


class EssentiaWrap70(QaWrapper):
    """
    Essentia Solution.
    """

    def compute(self, x):
        startstopsilence = es.StartStopSilence(threshold=-70)
        for frame in es.FrameGenerator(x, frameSize=2048, hopSize=512, startFromZero=True):
            result = startstopsilence(esarray(frame))
        solution = np.array([result[0], result[1]])
        return solution * 512 / 44100.

class LibrosaWrap(QaWrapper):
    """
    Librosa solution.
    """

    def compute(self, x):
        _, solution = trim(x)
        return solution / 44100.


class Distance(QaMetric):
    def score(self, reference, estimated):
        scores = np.abs(reference - estimated)
        return scores


if __name__ == '__main__':
    # Instantiating wrappers
    wrappers = [
        EssentiaWrap60('events'),
        EssentiaWrap70('events'),
        EssentiaWrap50('events'),
        LibrosaWrap('events', ground_true=True)  # May act as GT when annotations are not available
    ]

    # Instantiating the test
    qa = QaTestEvents()

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # Add the testing files
    qa.set_data(filename='../../QA-audio/StartStopSilence/')  # Works for a single

    # Add extra metrics
    qa.set_metrics(Distance())

    # Add ground true
    qa.load('../../QA-audio/StartStopSilence/', ground_true=True)

    # Compute and the results, the scores and and compare the computation times
    qa.compute_all()

    qa.plot_all(force=True, plots_dir='StartStopSilence/plots')

    qa.score_all()

    qa.generate_stats(output_file='StartStopSilence/stats.log')

    qa.compare_elapsed_times(output_file='StartStopSilence/stats.log')

    qa.save_test('StartStopSilence/test')
