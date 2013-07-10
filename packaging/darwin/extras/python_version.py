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



# this script is used in the preinstall stage to find out whether python2.5 is
# installed

import sys

def isCorrectVersion() :
    version = sys.version.split()[0].split('.')
    return int(version[0]) >= 2 and int(version[1]) >= 5


if __name__ == '__main__':
    if isCorrectVersion():
        sys.exit(0);
    else: sys.exit(1);
