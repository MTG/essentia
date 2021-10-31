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

import essentia
from essentia import EssentiaError, INFO
#import chords_dissonance_table


namespace = 'tonal'
dependencies = 'tonal'

circle_of_fifth = ['C','Em','G','Bm','D','F#m','A','C#m','E','G#m','B','D#m','F#','A#m','C#','Fm','G#','Cm','D#','Gm','A#','Dm','F','Am']

def get_chords_histogram_norm(chords, key):

    key_index = circle_of_fifth.index(key)

    chords_histogram = [0]*24
    if len(chords) > 0:
       for chord in chords:
           chord_index = circle_of_fifth.index(chord) - key_index
           if chord_index < 0:
              chord_index += len(circle_of_fifth)
           chords_histogram[chord_index] += 1

       for i in range(len(chords_histogram)):
           chords_histogram[i] *= 100.0 / len(chords)

    return chords_histogram


def get_chords_histogram(chords):

    chords_histogram = dict([(key, 0) for key in circle_of_fifth])

    for chord in chords:
        chords_histogram[chord] += 1.0

    for chord in chords_histogram.keys():
        chords_histogram[chord] *= 100.0 / len(chords)

    return chords_histogram


def compute(audio, pool, options):

    INFO('Computing Chords descriptors...')

    sampleRate = options['sampleRate']

    # get chords and key
    chords = pool.value('tonal.chords_progression')
    #chords_scope = pool.descriptors['tonal']['chords_progression']['scopes']
    key_key = pool.value('tonal.key_key')[0]
    key_scale = pool.value('tonal.key_scale')[0]
    if key_scale == 'minor':
       key = key_key + 'm'
    else:
       key = key_key

    # chords histogram
    chords_histogram_norm = get_chords_histogram_norm(chords, key)
    pool.add(namespace + '.' + 'chords_histogram', chords_histogram_norm)#, pool.GlobalScope)

    if len(chords) > 1:

       # chords number rate
       chords_number = 0
       for chord_value in chords_histogram_norm:
         if chord_value > 1.0:
            chords_number += 1
       chords_number_rate = float(chords_number) / len(chords)
       pool.add(namespace + '.' + 'chords_number_rate', chords_number_rate)#, pool.GlobalScope)

     # commented as at the moment pool has no scope
     #  # chord changes
     #  chords_changes = []
     #  for chord, chord_next, chord_scope in zip(chords[:-1], chords[1:], chords_scope):
     #       if chord != chord_next:
     #          chords_changes.append(chord_scope[0])
     #  pool.add('chords_changes', chords_changes, pool.GlobalScope)

       chords_changes = 0
       for chord, chord_next in zip(chords[:-1], chords[1:]):
            if chord != chord_next:
               chords_changes += 1
       # chord changes rate
       pool.add(namespace + '.' + 'chords_changes', chords_changes)
       chords_change_rate = float(chords_changes) / len(chords)
       pool.add(namespace + '.' + 'chords_changes_rate', chords_change_rate)#, pool.GlobalScope)

       # finding the key = the most frequent chord
       chords_histogram = get_chords_histogram(chords)
       # 1st step: find the most frequent chord(s)
       max_value = max(chords_histogram.values())
       chords_max = []
       for chord in chords_histogram.keys():
         if chords_histogram[chord] == max_value:
            chords_max.append(chord)
       # 2nd step: in case of 2 max, let's take the major one
       key = chords_max[0]
       if len(chords_max) > 1:
          for chord in chords_max:
            chord_split = chord.split("m")
            if len(chord_split) == 1:
               key = chord
       # 3rd step: fill the pool
       key_split = key.split("m")
       if len(key_split) == 1:
          chord_key = key_split[0]
          chord_scale = 'major'
       else:
          chord_key = key_split[0]
          chord_scale = 'minor'
       pool.add(namespace + '.' + 'chords_key', chord_key)#, pool.GlobalScope)
       pool.add(namespace + '.' + 'chords_scale', chord_scale)#, pool.GlobalScope)

    else:
       pool.add(namespace + '.' + 'chords_number_rate.undefined')#, pool.GlobalScope)
       pool.add(namespace + '.' + 'chords_changes.undefined')#, pool.GlobalScope)
       pool.add(namespace + '.' + 'chords_changes_rate.undefined')#, pool.GlobalScope)
       pool.add(namespace + '.' + 'chords_dissonance.undefined')#, pool.GlobalScope)
       pool.add(namespace + '.' + 'chords_key.undefined')#, pool.GlobalScope)
       pool.add(namespace + '.' + 'chords_scale.undefined')#, pool.GlobalScope)

    INFO('100% done...')
