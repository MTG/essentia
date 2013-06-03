#!/usr/bin/env python
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
