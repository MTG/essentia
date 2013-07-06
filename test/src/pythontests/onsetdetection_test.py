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

#! /usr/bin/python

# example script to compute and plot onsetdetection related descriptors

from essentia.extractor.onsetdetection import compute

def parse_args():
    from optparse import OptionParser
    import sys
    usage = "usage: %s [-v] <-i input_soundfile> [-g ground_truth_file]" % sys.argv[0]
    parser = OptionParser(usage=usage)
    parser.add_option("-v","--verbose",
        action="store_true", dest="verbose", default=False,
        help="verbose mode")
    parser.add_option("-i","--input",
        action="store", dest="input_file", type="string",
        help="input file")
    parser.add_option("-w","--wave-output", default=None,
        action="store", dest="wave_output", type="string",
        help="wave output filename")
    (options, args) = parser.parse_args()
    if options.input_file is None:
      print usage
      sys.exit(1)
    return options, args

if __name__ == '__main__':
    import sys, os.path, essentia
    options, args = parse_args()
    input_file = options.input_file

    # load audio file
    audio_file = essentia.AudioFileInput(filename = input_file)
    audio = audio_file()
    sampleRate = 44100.
    pool = essentia.Pool(input_file)

    compute(audio, pool, sampleRate = sampleRate, verbose = options.verbose)
    onsets = list(pool.descriptors['rhythm_onsets']['values'][0])
    if (options.verbose): print onsets
    while ((onsets[-1] + 0.020) * sampleRate) > len(audio):
      onsets.pop(len(onsets)-1)
    if len(onsets) > 0 and options.wave_output != None:
      #audio *= 0.# only ticks
      tick_length = 0.020 # in seconds
      for tick in onsets:
        for sample in range( int(round(tick*sampleRate)), int(round( (tick + tick_length ) *sampleRate )) ):
          audio[sample] += 0.4 * ( sample % 200. - 100. )
      output_file = essentia.WaveFileOutput(filename = options.wave_output)
      output_file(audio)

