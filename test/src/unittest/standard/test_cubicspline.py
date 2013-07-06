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

from essentia.standard import *
from numpy import r_, sin, pi # r_ for decimal step ranges


class TestCubicSpline(TestCase):

    def testBadParams(self):

        self.assertConfigureFails(CubicSpline(), { 'leftBoundaryFlag': 3})
        self.assertConfigureFails(CubicSpline(), { 'rightBoundaryFlag': -3})

        # ascendant order
        self.assertConfigureFails(CubicSpline(), { 'xPoints': [ 0, 10, 10, 20 ],
                                                   'yPoints': [ 0, 5, -23, 17 ] })

        # xPoints.size != yPoints.size
        self.assertConfigureFails(CubicSpline(), { 'xPoints': [ 0, 10, 10, 20, 30],
                                                   'yPoints': [ 0, 5, -23, 17 ] })

        # even sizes for quadratic spline should fail
        self.assertConfigureFails(CubicSpline(), { 'xPoints': [ 0, 10, 10, 20],
                                                   'yPoints': [ 0, 5, -23, 17 ] })

        self.assertConfigureFails(CubicSpline(), { 'xPoints': [ 10, 0 ],
                                                   'yPoints': [ 0, 10 ] })

    def runge(self, x): return 1.0/(1.0+25.0*float(x*x))

    def dRunge(self, x) :
        # Runge's first derivative at x
        k = 1.0+25.0*float(x*x)
        return -50.0*float(x)/(k*k)

    def ddRunge(self, x):
         # Runge's second derivative at x
         xx = x*x
         k = 1.0+25.0*float(xx)
         return (-50.0+3750.0*float(xx))/(k*k*k)

    def evaluateCubicSpline(self, expected, bc, plot=False):
        n = 11
        x =  [(float(n-i)*(-1.0)+float(i-1))/float(n-1) for i in range(n)]
        y =  [self.runge(i) for i in x]

        # just for plotting
        real=[];newx=[];found=[]

        if not bc:
            leftBound = 0.0 ;rightBound = 0.0
        elif bc == 1:
             leftBound = self.dRunge(x[0])
             rightBound = self.dRunge(x[-1])
        else: # bc == 2
             leftBound = self.ddRunge(x[0])
             rightBound = self.ddRunge(x[-1])

        spline = CubicSpline(xPoints=x,yPoints=y,
                             leftBoundaryFlag = bc,
                             leftBoundaryValue = leftBound,
                             rightBoundaryFlag = bc,
                             rightBoundaryValue = rightBound)
        xval = 0
        k=0
        for i in range(n+1):
            if not i: jhi = 1
            else: jhi = 2

            for j in range(1, jhi+1):
                if not i: xval = x[0] - 1.0
                elif i<n:
                    xval = (float(jhi-j+1)*x[i-1]+float(j-1)*x[i])/float(jhi)
                else:
                    if j==1: xval=x[n-1]
                    else: xval=x[n-1]+1.0
                yval = spline(xval)[0]
                self.assertAlmostEqual(expected[k], yval, 5e-6)
                newx.append(xval)
                found.append(yval)
                real.append(self.runge(xval))
                k+=1
        if plot:
            from pylab import plot, show, legend
            plot(newx, found, label='found')
            plot(newx, expected, label='expected')
            plot(newx, real, label='real')
            legend()
            show()

    def testCubicSplineBc0(self):
        expected = readVector(join(filedir(), 'spline/cubicSpline_bc0.txt'))
        self.evaluateCubicSpline(expected, 0)

    def testCubicSplineBc1(self):
        expected = readVector(join(filedir(), 'spline/cubicSpline_bc1.txt'))
        self.evaluateCubicSpline(expected, 1)

    def testCubicSplineBc2(self):
        expected = readVector(join(filedir(), 'spline/cubicSpline_bc2.txt'))
        self.evaluateCubicSpline(expected, 2)



suite = allTests(TestCubicSpline)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
