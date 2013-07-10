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



import os
import sys
import essentia
import yaml
import numpy
from essentia.extractor.extractor import compute

def clean_key(descriptors):
    key_mapping = {}
    key_mapping['A'] = 0
    key_mapping['A#'] = 1
    key_mapping['B'] = 2
    key_mapping['C'] = 3
    key_mapping['C#'] = 4
    key_mapping['D'] = 5
    key_mapping['D#'] = 6
    key_mapping['E'] = 7
    key_mapping['F'] = 8
    key_mapping['F#'] = 9
    key_mapping['G'] = 10
    key_mapping['G#'] = 11

    key_mapping['major'] = 1
    key_mapping['minor'] = 0

    # changing the key scale into a binary value
    descriptors['key_scale']['value'] = key_mapping[ descriptors['key_scale']['value'] ]

    # removing the key
    descriptors['key_key']['value'] = key_mapping[ descriptors['key_key']['value'] ]

def isAlmostEqual(a, b, precision):
    error = 0.0
    if (b == 0.0): error = abs(a-b)
    else: error = abs((a-b)/b)
    if error <= precision: return True
    else: return False

input_file = '../../../../audio/recorded/britney.wav'
output_file = input_file.replace(".wav",".sig")

# Python extractor
#print "Python extractor"

try:
  #possible also but i gave up :-)
  #pool_python = compute(input_file, True, True)
  #desc_python = pool_python.aggregate_descriptors()
  if sys.platform == 'win32':
    python_bin = 'c:\\Python24\\python.exe'
  else:
    python_bin = 'python'
  os.system(python_bin + ' ../../../src/python/essentia/extractor/essentia_music.py ' + input_file + ' ' + output_file)
  desc_python = yaml.load(open(output_file).read())
except (essentia.EssentiaError, RuntimeError):
  print 'ERROR:', sys.exc_type, sys.exc_value
  sys.exit(1)

clean_key(desc_python)
del desc_python['version']

# C++ extractor
#print
#print "C++ extractor"

try:
  audio_file = essentia.AudioFileInput(filename = input_file)
  audio = audio_file()
  essentia_music = essentia.EssentiaMusic(filename = output_file, mode = "music", verbose = False)
  essentia_music(audio)
  desc_c = yaml.load(open(output_file).read())
except (essentia.EssentiaError, RuntimeError):
  print 'ERROR:', sys.exc_type, sys.exc_value
  sys.exit(1)

os.remove(output_file)

# Comparation
#print
for desc in desc_c:
    if desc not in desc_python:
        print "ERROR: the descriptor " + desc + " is not included in the python version"
        sys.exit(1)
for desc in desc_python:
    if desc not in desc_c:
        print "ERROR: the descriptor " + desc + " is not included in the C++ version"
        sys.exit(1)

    if 'mean' in desc_python[desc] and type(desc_python[desc]['mean']) != list:
      desc_python[desc]['mean'] = [desc_python[desc]['mean']]

    if 'value' in desc_python[desc] and type(desc_python[desc]['value']) != list:
      desc_python[desc]['value'] = [desc_python[desc]['value']]

    if type(desc_c[desc]['mean']) != list:
      desc_c[desc]['mean'] = [desc_c[desc]['mean']]

    i = 0
    for (i,(value_python, value_c)) in enumerate(zip(desc_python[desc]['mean'], desc_c[desc]['mean'])):
        if not isAlmostEqual(value_python, value_c, 10e-4):
            print "ERROR: In descriptor ",desc,", value # ",i
            print "Python value =", value_python, "while C++ value =", value_c
            sys.exit(1)
        i+=1
