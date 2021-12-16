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


import sys
import essentia
from essentia import EssentiaError
from gaia2 import *
from math import *

def print_onset(onset):

    if onset <= 60.0:
       minutes = 0
       seconds = int(floor(onset))
    else:
       minutes = int(floor(onset / 60.0))
       seconds = int(floor(onset % 60))
    print minutes, "mn", seconds, "s",


def compute(megalopool, verbose = True):

    cvar.verbose = False

    if verbose:
       print "\nDoing thumbnailing..."

    # From megalopool to Gaia point
    p = megalopool.to_point()
    p_name = 'song'
    p.setName(p_name)
    nsegs = p.numberSegments()

    # Creating dataset containing different segments
    dataset = DataSet()
    c = dataset.addCollection('segments')
    for i in range(1, nsegs):
      c.addPoint(Point.fromSingleSegment(p, i))

    # Transforming the dataset
    to_select = [
      'loudness_replay_gain.value',
      'spectral_centroid.mean',
      'spectral_complexity.mean',
      'spectral_crest.mean',
      'spectral_decrease.mean',
      'spectral_flux.mean',
      'spectral_hfc.mean',
      'spectral_pitch_histogram_spread.value',
      'spectral_pitch_instantaneous_confidence.mean',
      'spectral_silence_rate_30dB.mean',
      'temporal_zerocrossingrate.mean',
    ]
    dataset_select = transform(dataset, 'select', {'descriptorNames': to_select})
    dataset_clean = transform(dataset_select, 'cleaner')
    dataset_norm = transform(dataset_clean, 'normalize')

    # Creating view
    distance = DistanceFunctionFactory.create('euclidean', dataset_norm.layout())
    view = View(dataset_norm, distance)

    # NN-search
    nn_number = 3
    if nn_number > nsegs -1:
       nn_number = nsegs -1
    segment_distance_min = 100000.0
    thumbnail_name = ''
    for i in range(1, nsegs):
      point_name = Point.fromSingleSegment(p, i).name()
      point_norm = dataset_norm.point(point_name)
      segment_list = view.nnSearch(point_norm, nn_number)
      segment_distance = 0
      for j in range(1, nn_number):
        segment_distance += segment_list[j][1]
      if segment_distance < segment_distance_min:
         segment_distance_min = segment_distance
         thumbnail_name = point_name

    thumbnail_new_name = thumbnail_name.replace(p_name + '_', '')
    thumbnail = megalopool.segments[thumbnail_new_name]['scope']

    if verbose:
       if len(thumbnail) > 0:
          print 'Thumbnail : ',
          print '[',
          print_onset(thumbnail[0])
          print ",",
          print_onset(thumbnail[1])
          print '] ',
       else:
          print 'No thumbnail found!'
       print

    return thumbnail
