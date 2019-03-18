#!/usr/bin/env python

# Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

import essentia.standard as es

import pyloudness

sys.path.insert(0, './')
from qa_test import *
from qa_testvalues import QaTestValues


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.LoudnessEBUR128()

    def compute(self, *args):
        outs = self.algo(args[1])

        integratedLoudness = outs[2]
        loudnessRange = outs[3]

        return esarr([integratedLoudness, loudnessRange])


class PyloudnessWrap(QaWrapper):
    """
    Pyloudness Solution. Found on this repo https://github.com/jrigden/pyloudness
    Just a wrapper for the FFMPEG pure c implementation
    https://github.com/FFmpeg/FFmpeg/blob/ed93ed5ee320db299dc8c65d59c4f25e2eb0acdc/libavfilter/ebur128.h
    """
    def compute(self, *args):
        key = args[2]
        outs = pyloudness.get_loudness(args[0].routes[key])

        integratedLoudness = outs['Integrated Loudness']['I']
        loudnessRange = outs['Loudness Range']['LRA']

        return esarr([integratedLoudness, loudnessRange])


if __name__ == '__main__':
    # We are using 1 digit only to fit the format of PyloudnessWrap
    np.set_printoptions(precision=1)

    # Instantiating wrappers
    wrappers = [
        EssentiaWrap('values'),
        PyloudnessWrap('values', ground_true=True),
    ]

    # Instantiating the test
    qa = QaTestValues(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # Add the testing files
    qa.load_audio(filename='../../audio/recorded/',
                  stereo=True)  # Works for a single

    # Compute and the results, the scores and and compare the computation times

    qa.compute_all()

    qa.score_all()
