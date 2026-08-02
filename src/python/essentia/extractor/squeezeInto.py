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



from numpy import tanh

def squeezeIntoZeroToOne(x1,x2,x):
    '''This function returns values between zero and one
       with large negative x-values yielding zero and large
       positive x-values yielding one. x-value between x1
       and x2 are mapped to the range of 0.119 to 0.881.
    '''
    return squeezeInto([x1,0],[x2,1],x)
    #this is the same as return 0.5+0.5*tanh(-1+2*((x-x1)/(x2-x1)))

def squeezeInto(p1,p2,x):
    '''This function returns values between p1[1] and p2[1]
       with large negative x-values yielding zero and large
       positive x-values yielding one.
    '''
    if p2[0] > p1[0]:
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]
    else:
        x1 = p2[0]
        x2 = p1[0]
        y1 = p2[1]
        y2 = p1[1]
    return y1 + (y2 - y1) * (0.5+0.5*tanh(-1+2*((x-x1)/(x2-x1))))


#testing
'''
import pylab
pylab.clf()
x = []
y1 = []
y2 = []
N = 100
p1 = [-15, -1]
p2 = [-3, 5]
x0 = p1[0] - (p2[0] - p1[0])
x3 = p2[0] + (p2[0] - p1[0])
for i in range(N):
    x.append(x0+1.0*i/N*(x3-x0))
    y1.append(squeezeIntoZeroToOne(p1[0],p2[0],x[i]))
    y2.append(squeezeInto(p1,p2,x[i]))
pylab.plot([p1[0],p2[0]],[p1[1],p2[1]],'.r')
pylab.plot(x,y1)
pylab.plot(x,y2)
#pylab.ylim([y1-0.1,y2+0.1])
pylab.show()
'''
