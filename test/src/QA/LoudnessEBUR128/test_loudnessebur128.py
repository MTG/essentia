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
from qa_testvalues import QaTestValues
import essentia.standard as es

import pyloudness


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    def compute(self, *args):

        # stereo = es.StereoMuxer()(args[1], args[1])
        outs = es.LoudnessEBUR128()(args[1])

        integratedLoudness = outs[2]
        loudnessRange = outs[3]

        return esarr([integratedLoudness, loudnessRange])


class PyloudnessWrap(QaWrapper):
    """
    Pyloudness Solution. Found on this repo https://github.com/jrigden/pyloudness
    I didn't find any feedback about the lib and the developer doesen't seem to have
    a huge background in audio processing.
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
    data_dir = '../../../../../../pablo/Music/Desakato-La_Teoria_del_Fuego/'
    # data_dir = '../../audio/recorded'
    # qa.load_audio(filename='../../QA-audio/Jumps/prominent_jumps')  # Works for a single
    # qa.load_audio(filename='../../../../../data/Dead_Combo_-_01_-_Povo_Que_Cas_Descalo_silence.wav')  # Works for a single
    #  qa.load_audio(filename='../../QA-audio/Jumps/loud_songs/')  # Works for a single

    qa.load_audio(filename=data_dir, stereo=True)  # Works for a single

    # qa.load_solution(data_dir, ground_true=True)

    # Compute and the results, the scores and and compare the computation times

    qa.compute_all()

    qa.score_all()
