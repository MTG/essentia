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



import yaml
import sys
import os
from pylab import *

results_path = sys.argv[1]

tree = os.walk(results_path)

genre_map = {
             'Alternative':'Pop-Rock',
             'Blues':'Blues',
             'Classical':'Classical',
             'Country':'Country',
             'Dance and House':'Electro',
             'Folk and New Age':'Folk and New Age',
             'Hip-Hop and Rap':'Hip-Hop and Rap',
             'House & Garage and Grime':'Electro',
             'Jazz':'Jazz',
             'Jungle and D&B':'Electro',
             'Latin':'Latin',
             'Pop':'Pop-Rock',
             'R&B and Soul':'R&B and Soul',
             'Reggae':'Reggae',
             'Rock and Metal':'Metal-Punk',
             'Sample':'Sample',
             'Techno and Electro':'Electro',
             'Vocal and Acapella':'Vocal and Acapella'
            }

eq_genres = {}
number_genres = {}

for root, dirs, files in tree:
    if '.svn' in dirs:
        dirs.remove('.svn')
    for filename in files:

          genre = genre_map[root.split('/')[-1]]
          #if genre == "Techno and Electro" or genre == "Classical":

          results_file = os.path.join(root, filename)

          print 'Processing file:'
          print results_file

          y = yaml.load( open(results_file).read(), yaml.CLoader )

          bass = y['energybandratio_bass']['mean']
          middle_low = y['energybandratio_middle_low']['mean']
          middle_high = y['energybandratio_middle_high']['mean']
          high = y['energybandratio_high']['mean']

          if not genre in eq_genres:
             eq_genres[genre] = { 'bass':0, 'middle_low':0, 'middle_high':0, 'high':0 }
          eq_genres[genre]['bass'] += bass
          eq_genres[genre]['middle_low'] += middle_low
          eq_genres[genre]['middle_high'] += middle_high
          eq_genres[genre]['high'] += high

          if not genre in number_genres:
             number_genres[genre] = 0
          number_genres[genre] += 1
          print

# normalization
for genre in eq_genres.keys():
    for band in eq_genres[genre].keys():
        eq_genres[genre][band] /= number_genres[genre]

print eq_genres



i=1
html = '<html>'
for genre in eq_genres.keys():
   if genre != 'Sample':
      figure(i)
      eq_genre = []
      eq_genre.append(eq_genres[genre]['bass'])
      eq_genre.append(eq_genres[genre]['middle_low'])
      eq_genre.append(eq_genres[genre]['middle_high'])
      eq_genre.append(eq_genres[genre]['high'])
      bar(range(4), eq_genre)
      axis([0, 4, 0, 0.7])
      figfile = genre + '.png'
      savefig( figfile, dpi=60, format='png')
      html += '<div>'
      html += '<h2>%s</h2>' % genre
      html += '<img src="%s" />' % figfile
      html += '</div>'
      i+=1
html += '</html>'

out = open('index.html', 'w')
out.write(html)
