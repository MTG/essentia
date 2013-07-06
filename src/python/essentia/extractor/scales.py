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

noteNumbers = { 'Cb':11, 'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'E#':5, 'Fb':4, 'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11, 'B#':0 }

majorScale = [ 2, 2, 1, 2, 2, 2, 1 ]

minorHarmonicScale = [ 2, 1, 2, 2, 1, 3, 1 ]

majorChromaticity = [0, 1, 0, 1, 0, 0, 0.5, 0, 1, 0, 0.5, 0]

minorHarmonicChromaticity = [0, 1, 0, 0, 1, 0, 0.5, 0, 0, 1, 0.5, 0]

def makeChromaticWeights( chromaticity, tonic, octaves ):
    weights = [0]*tonic
    weights += chromaticity*octaves
    return weights

def makeScale( scaleSteps, tonic ):
    n = tonic
    scale = [tonic]
    for s in scaleSteps[:-1]:
	n += s
	scale.append(n)
    return scale

def makeFullScale( scaleSteps, tonic, octaveBegin, octaveEnd ):
    s = makeScale( scaleSteps, tonic )
    scale = []
    i = 0
    octave = octaveBegin
    while octave <= octaveEnd:
	scale.append( octave*12+s[i] )
	i += 1
	if i == len(s):
	    i = 0
	    octave += 1

    return scale
