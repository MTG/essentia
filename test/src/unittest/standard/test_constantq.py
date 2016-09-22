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



from essentia_test import *
import math
import numpy as np
import os

script_dir = os.path.dirname(__file__)


def cvec(l):
   	return numpy.array(l, dtype='c8')

class TestCQ(TestCase):

    
    def testRandom(self):
        # input is [1, 0, 0, ...] which corresponds to an ConstantQ of constant magnitude 1
        with open(os.path.join(script_dir,'constantq/CQinput.txt'), 'r') as f:
        	#read_data = f.read()
        	data = np.array([], dtype='complex64')
        	line = f.readline()
        	while line != '':
        		re = float(line.split('\t')[0])
        		im = float(line.split('\t')[1])
        		data = np.append(data, re + im * 1j)
        		line = f.readline()
				

        with open(os.path.join(script_dir,'constantq/QMoutput.txt'), 'r') as u:
            QMdata_out = np.array([], dtype='complex64')        
            for line in u:
                re = line.split('+')[0]
                re = float(re[1:])
                im = line.split('+')[1]
                im = float(im[:-3])
                QMdata_out = np.append(QMdata_out, re + im * 1j)
       

        CQdata = ConstantQ()(cvec(data))
        QMdata_out = np.array(QMdata_out, dtype='complex64') 
        

        DifferMean = QMdata_out-CQdata  # difference mean
        DifferMean = ((sum(abs(DifferMean.real))/len(DifferMean))+(sum(abs(DifferMean.imag))/len(DifferMean)))/2 # difference mean
        DiverPerReal = (sum(((QMdata_out.real-CQdata.real)/QMdata_out.real)*100))/len(QMdata_out.real) #divergence mean percentage 
        DiverPerImag = (sum(((QMdata_out.imag-CQdata.imag)/QMdata_out.imag)*100))/len(QMdata_out.imag)  #divergence mean percentage 
        DiverPer = (DiverPerReal + DiverPerImag) / 2

        
        """
        print 'Divergence mean percentage is : '+str(DiverPer)[:5]+'%'
        print 'Difference Mean is : '+str(DifferMean)[:9]
        """



suite = allTests(TestCQ)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

