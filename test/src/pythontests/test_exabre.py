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


import numpy
import essentia
import glob
import essentia
from essentia.extractor.extractor import compute

input_files = glob.glob("../../../../audio/recorded/*.wav")

for input_file in input_files:

    pool = compute(input_file, True, True)
    desc = pool.aggregate_descriptors()

    key_strength = desc['key_strength']['mean']

    if key_strength <= 0.33:
       mel = 1
    else:
       if (key_strength > 0.33) and (key_strength <= 0.66):
          mel = 2
       else:
          mel = 3

    print
    print "Melodicness Python = ", ton

    exabre = essentia.Exabre()
    ton_c = exabre(float(key_strength))
    print "Melodicness C = ", ton
    print

output.close()
