#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

class TestHighPass(TestCase):

    def testRegression(self):
        sr = 44100.
        pi2 = 2*pi
        signal = [.25*cos(t*pi2*5/sr) + \
                  .25*cos(t*pi2*50/sr) + \
                  .25*cos(t*pi2*500./sr) + \
                  .25*cos(t*pi2*5000./sr)
                  for t in range(44100)]

        filteredSignal = HighPass(cutoffFrequency=1000)(signal)

        s = Spectrum()(signal)
        sf = Spectrum()(filteredSignal)

        for i in range(1000):
            if s[i] > 10:
                self.assertTrue(sf[i] < 0.5*s[i])

        for i in range(1001, len(s)):
            if s[i] > 10:
                self.assertTrue(sf[i] > 0.5*s[i])


suite = allTests(TestHighPass)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
