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

#!/usr/bin/python

import sys
import logging

class Progress:
    def __init__(self, total, format = '%d%% done...'):
        if total < 0:
            raise IndexError('total needs to be >= 0')
        self.total = total
        self.current = 0
        self.format = '\r' + format

        # small hack: if total = 0, then we're already done, so pretend we
        # actually did something and just print 100%
        if total == 0:
            self.total = 1
            self.current = 1

    def percent(self, n):
        return n*100 / self.total

    def verbose(self):
        return logging.getLogger().isEnabledFor(logging.INFO)

    def update(self, current):
        shouldUpdate = (self.percent(current) > self.percent(self.current))
        self.current = current
        if shouldUpdate:
            self.updateDisplay()

    def updateDisplay(self):
        if self.verbose():
            print self.format % self.percent(self.current),
            sys.stdout.flush()

    def finish(self):
        self.update(self.total)
        # also account for the final end of line
        if self.verbose():
            print
