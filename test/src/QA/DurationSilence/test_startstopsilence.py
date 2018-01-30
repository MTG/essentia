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
import numpy as np
from librosa.effects import trim
import essentia.standard as es
from essentia import array as esarray


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    def compute(self, x):
        startstopsilence = es.StartStopSilence()
        for frame in es.FrameGenerator(x, frameSize=2048, hopSize=512, startFromZero=True):
            result = startstopsilence(esarray(frame))
        solution = np.array([result[0], result[1]])
        return solution * 512 / 44100.


class FrameGeneratorWrap(QaWrapper):
    """
    Essentia frame-wise analysis to assess the time consumed by StartStopSilence itself.
    """
    def compute(self, x):
        import essentia.standard as es
        startstopsilence = es.StartStopSilence()

        for frame in es.FrameGenerator(x, frameSize=2028, hopSize=512, startFromZero=True):
            result = frame[:2]
        solution = np.array([0.1, 1.1]) * 512 / 44100.
        return solution


class LibrosaWrap(QaWrapper):
    """
    Librosa solution.
    """
    def compute(self, x):
        _, solution = trim(x)
        return solution / 44100.


""" 
Not used because it does not output the start/end index
class MadmonWrap(QaWrapper):

    def compute(self, x):
        from madmom.audio.signal import trim as trim_madmom
        _, solution = trim_madmom(x)
        return solution / 44100.
"""


class MirEvalOnsetWrap(QaMetric):
    """
    mir_eval metrics wrapper. More information in https://craffel.github.io/mir_eval/#module-mir_eval.onset
    """
    def score(self, reference, estimated):
        import mir_eval as me
        scores = me.onset.evaluate(reference, estimated)
        return scores


if __name__ == '__main__':
    # wrappers
    wrappers = [
        EssentiaWrap('events'),
        FrameGeneratorWrap('events'),
        LibrosaWrap('events', ground_true=True)
    ]

    # test itself
    qa = QaTest('events')

    # Set the items
    qa.set_wrappers(wrappers)

    # Works for a single track or a folder
    #  qa.set_data(filename='../../audio/recorded/Vivaldi_Sonata_5_II_Allegro.wav')
    qa.set_data(filename='../../audio/recorded/')

    qa.set_metrics(MirEvalOnsetWrap())

    # Compute and scoring
    qa.compute_all()
    qa.score_all()

    qa.compare_elapsed_times()
