#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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



from essentia_test import *
from math import *
from numpy import *

from essentia import *
from math import *
import essentia.standard as std


class TestMaxFilter(TestCase):

    def testRegression(self):
      sr = 44100
      index=0
      original_signal = []
      clean_signal = []
      while index < 1000:
        original_signal_pt = .25 * cos((index/sr)  * 5 * 2*pi) \
                            +.25 * cos((index/sr)  * 50 * 2*pi) \
                            +.25 * cos((index/sr)  * 500 * 2*pi) \
                            +.25 * cos((index/sr)  * 5000 * 2*pi)
        clean_signal.append(original_signal_pt)
        index+=1

      maxfilteredSignal = std.MaxFilter()(clean_signal)

      smf = std.Spectrum()(maxfilteredSignal)
      print("smf[11]")
      print(smf[11])
      print("smf[113]")
      print(smf[113])
      print("smf[227]")
      print(smf[227])

suite = allTests(TestMaxFilter)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
