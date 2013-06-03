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
