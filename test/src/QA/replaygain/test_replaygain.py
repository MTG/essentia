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
import subprocess

import essentia.standard as es

sys.path.insert(0, './')
from qa_test import *
from qa_testvalues import QaTestValues


class EssentiaWrap(QaWrapper):
    """
    Essentia Solution.
    """
    algo = es.ReplayGain()

    def compute(self, *args):
        y = self.algo(args[1])

        return esarr(y)


class RGain(QaWrapper):
    """
    rgain Solution. Found on this repo https://pypi.python.org/pypi/rgain
    rgain relies on GStream tools for the analysis of the replay gain
    According to this pagem the reference level should be set to 83 for
    the case of 'no preamplification' (as it is implemented in Essentia).
    https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-good/html/gst-plugins-good-plugins-rganalysis.html#GstRgAnalysis--reference-level
    """
    def compute(self, *args):
        key = args[2]
        command = ['replaygain', '-f', '-d', '-r 83', args[0].routes[key]]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT,
                                         universal_newlines=True)
        if output.endswith('Nothing to do.'):
            y = float(output.split('\n')[1].split(' ')[-2])
        else:
            return esarr(np.nan)

        return esarr(y)


class bs1770gain(QaWrapper):
    """
    This program implements ReplayGain 2.0 an should not be considered in the comparison. The wrapper was implemented
    just to assess the difference among versions. To install it follow : http://bs1770gain.sourceforge.net/
    """
    def compute(self, *args):
        key = args[2]
        command = ['bs1770gain', '--replaygain', args[0].routes[key]]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT,
                                         universal_newlines=True)
        y = float(output.split('\n')[2].split(' ')[-5]) -6.

        return esarr(y)


if __name__ == '__main__':
    folder = 'replaygain'

    # We are using 1 digit only to fit the format of PyloudnessWrap
    np.set_printoptions(precision=1)

    # Instantiating wrappers
    wrappers = [
        EssentiaWrap('values'),
        RGain('values', ground_true=True),
        bs1770gain('values')
    ]

    # Instantiating the test
    qa = QaTestValues(verbose=True)

    # Add the wrappers to the test the wrappers
    qa.set_wrappers(wrappers)

    # Add the testing files
    data_dir = '../../audio/recorded'

    qa.load_audio(filename=data_dir, stereo=False)  # Works for a single

    # Compute and the results, the scores and and compare the computation times
    qa.compute_all(output_file='{}/compute.log'.format(folder))
