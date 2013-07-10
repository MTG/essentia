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

##############################################################################
#
# arff2matlab.py - Writes a Matlab file from a weka source. Vectors are stored
#in rows of a matrix M, descriptors as columns.
#
# Class stuff not implemented yet.
#
##############################################################################

import sys
import string

##############################################################################

def comprovate(ok):
	if ok==0:
		print('Wrong input file. Not ARFF format or incorrect match with tags.\n')
		sys.exit()
	return 0

##############################################################################

fn_in=sys.argv[1]
fn_out=sys.argv[2]

tag1='@attribute'
tag1B='numeric'
tag2='@data'

fin=open(fn_in,'r')

# find '@attribute'
ok=comprovate(1)
while 1:
	line=fin.readline()
	if line=='': break
	line=line[0:len(line)-1]
	if tag1 in line:
		ok=1
		break

# save attributes and find data
ok=comprovate(ok)
attributes=[]
i=1
if line[len(tag1)+i]=='\'': i=2
attributes.append(str(line[len(tag1)+i:len(line)-len(tag1B)-i]))
while 1:
	line=fin.readline()
	if line=='': break
	line=line[0:len(line)-1]
	if tag2 in line:
		ok=1
		break
	if tag1 in line:
		i=1
		if line[len(tag1)+i]=='\'': i=2
		attributes.append(str(line[len(tag1)+i:len(line)-len(tag1B)-i]))

# save data
ok=comprovate(ok)
M=[]
while 1:
	line=fin.readline()
	if line=='': break
	line=line[0:len(line)-1]
	fields=line.split(',')
	if len(fields)!=0:
		M.append(fields)
		ok=1
	ok=comprovate(ok)

fin.close()


fout=open(fn_out,'w')

# write attributes
fout.write('attr={')
for attr in attributes:
	fout.write('\''+attr+'\' ')
fout.write('};\n')

# write matrix
fout.write('M=[')
for vector in M:
	for item in vector:
		fout.write(str(item)+' ')
	fout.write(';\n')
fout.write('];\n')

fout.close()

#############################################################################
