#!/usr/bin/env python

import yaml
import sys
import os
from pylab import *

results_path = sys.argv[1]

chords = { 'C':1, 'G':3, 'D':5, 'A':7, 'E': 9, 'B':11, 'F#':13, 'C#':15, 'G#':17, 'D#':19, 'A#':21, 'F':23,
           'Am':2, 'Em':4, 'Bm':6, 'F#m':8, 'C#m':10, 'G#m':12, 'D#m':14, 'A#m':16, 'Fm':18, 'Cm':20, 'Gm':22, 'Dm':24 }

tree = os.walk(results_path)

#histo = [0]*24
distances = {}
correct_key = 0
false_key = 0

for root, dirs, files in tree:
    if '.svn' in dirs:
        dirs.remove('.svn')
    for filename in files:

        genre = root.split('/')[-1]
        results_file = os.path.join(root, filename)

        print 'Processing file:'
        print results_file

        y = yaml.load( open(results_file).read(), yaml.CLoader )

        print y['chords']['values']
        
        key = y['key_key']['value']
        scale = y['key_scale']['value']

        if scale=='minor':
            key += 'm'

        key_hist = [0]*25
        for c in y['chords']['values']:
            if c == 'S' or c == 'U':
                continue
            key_hist[chords[c.strip('m')]] += 1
            mod = chords[c] % chords[key]
            print mod, ' ',
            if not genre in distances:
                distances[genre] = [0]*25
            #distances[genre].append( mod )
            distances[genre][mod] += 1
            #histo[mod] += 1

        calc_key = key_hist.index( max(key_hist) )
        if calc_key == chords[key.strip('m')]:
            correct_key += 1
        else:
            false_key += 1
        
        print

#print histo

i=1
html = '<html>'
for genre, distance in distances.items():
    figure(i)
    bar( range(25), distance )
    #n, bins, patches = hist( distance, 24 )
    #xlabel('values')
    #ylabel('files')
    #title( stat )
    figfile = genre + '.png' 
    savefig( figfile, dpi=60, format='png')
    i+=1

    html += '<div>'
    html += '<h2>%s</h2>' % genre
    html += '<img src="%s" />' % figfile
    html += '</div>'

html += '<p>Correct key guesses: %d</p>' % correct_key
html += '<p>Incorrect key guesses: %d</p>' % false_key
html += '</html>'

out = open( 'index.html', 'w' )
out.write( html )
