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

sys.path.insert(0, './')
from qa_test import *
from qa_testvalues import QaTestValues


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """

    def compute(self, *args):
        algo = es.HumDetector(minimumDuration=10.)

        x = args[1]
        _, frequencies, saliences, starts, ends = algo(x)

        return frequencies


if __name__ == '__main__':
    folder = 'humdetector'

    # Instantiating wrappers
    wrappers = [
        EssentiaWrap('events'),
    ]

    # Instantiating the test
    qa = QaTestValues(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # Do this with music from LaCupula
    data_dir = '../../QA-audio/Hum/Songs50HzHum'

    qa.load_audio(filename=data_dir, stereo=False)

    qa.compute_all(output_file='{}/compute.log'.format(folder))

    print(qa.solutions)

    GT = 50  # Hz
    detected = 0
    error = []
    for i in qa.solutions.values():
        if i.any():
            detected += 1
            error.append(np.mean(np.abs(i - GT)))

    print('Mean error: {:.2f}Hz'.format(np.mean(error)))
    print('Detected on {:.2f}% of the songs'.format(100. * detected /
                                                    len(qa.solutions.items())))
