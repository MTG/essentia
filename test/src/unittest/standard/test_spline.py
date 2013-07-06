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


class TestSpline(TestCase):

    def testBadParams(self):

        self.assertConfigureFails(Spline(), { 'type': 'hermite'})

        self.assertConfigureFails(Spline(), { 'type': 'beta',
                                              'beta1': -1 })

        self.assertConfigureFails(Spline(), { 'type': 'beta',
                                              'beta2': -1 })

        # ascendant order
        self.assertConfigureFails(Spline(), { 'xPoints': [ 0, 10, 10, 20 ],
                                              'yPoints': [ 0, 5, -23, 17 ] })

        # xPoints.size != yPoints.size
        self.assertConfigureFails(Spline(), { 'xPoints': [ 0, 10, 10, 20, 30],
                                               'yPoints': [ 0, 5, -23, 17 ] })

        # even sizes for quadratic spline should fail
        self.assertConfigureFails(Spline(), { 'xPoints': [ 0, 10, 10, 20],
                                              'yPoints': [ 0, 5, -23, 17 ] })

        self.assertConfigureFails(Spline(), { 'xPoints': [ 10, 0 ],
                                              'yPoints': [ 0, 10 ] })


    def evaluateBSpline(self, expected, plot=False):
       n = 11
       nsample = 4
       x = [float(i) for i in range(n)]
       y = [sin(2.0*pi*float(i)/float(n-1)) for i in x]
       found = []
       newx = []
       bspline = Spline(xPoints=x,yPoints=y, type='b')

       k=0
       for i in range(n+1):
           if not i:
               tlo = x[0]-0.5*(x[1]-x[0])
               thi = x[0]
               jhi = nsample-1
           elif i < n:
               tlo = x[i-1]
               thi = x[i]
               jhi = nsample-1
           elif i >= n:
               tlo = x[n-1]
               thi = x[n-1] + 0.5*(x[n-1]-x[n-2])
               jhi = nsample
           for j in range(jhi+1):
               xval=((nsample - j)*tlo+j*thi)/float(nsample)
               newx.append(xval)
               yval = bspline(xval)
               found.append(yval)
               # when the value is very small the precision error goes pretty
               # high up to 0.5, we skip them ... this is probably due to the
               # double to real conversion in essentia as the expected values were
               # computed with double precission
               if yval > 5e-16: self.assertAlmostEqual(expected[k], yval, 5e-6)
               k+=1
       if plot:
           from pylab import plot, show, legend
           plot(newx, found, label='found')
           plot(newx, expected, label='expected')
           legend()
           show()

    def evaluateBetaSpline(self, beta1, beta2, expected, plot=False):
       n = 11
       nsample = 4
       x = [float(i) for i in range(n)]
       y = [sin(2.0*pi*float(i)/float(n-1)) for i in x]
       found = []
       newx = []
       betaspline = Spline(xPoints=x,yPoints=y,beta1=beta1,beta2=beta2, type='beta')

       k=0
       for i in range(n+1):
           if not i:
               tlo = x[0]-0.5*(x[1]-x[0])
               thi = x[0]
               jhi = nsample-1
           elif i < n:
               tlo = x[i-1]
               thi = x[i]
               jhi = nsample-1
           elif i >= n:
               tlo = x[n-1]
               thi = x[n-1] + 0.5*(x[n-1]-x[n-2])
               jhi = nsample
           for j in range(jhi+1):
               xval=((nsample - j)*tlo+j*thi)/float(nsample)
               newx.append(xval)
               yval = betaspline(xval)
               found.append(yval)
               # when the value is very small the precision error goes pretty
               # high up to 0.5, we skip them ... this is probably due to the
               # double to real conversion in essentia as the expected values were
               # computed with double precission
               if yval > 5e-16: self.assertAlmostEqual(expected[k], yval, 5e-6)
               k+=1
       if plot:
           from pylab import plot, show, legend
           plot(newx, found, label='found')
           plot(newx, expected, label='expected')
           legend()
           show()

    def runge(self, x): return 1.0/(1.0+25.0*float(x*x))

    def evaluateQuadraticSpline(self, expected, plot=False):
        n = 11
        x =  [(float(n-i)*(-1.0)+float(i-1))/float(n-1) for i in range(n)]
        y =  [self.runge(i) for i in x] # Runge function
        qspline=Spline(xPoints=x,yPoints=y, type='quadratic')
        real=[]
        newx=[]
        found=[]

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
                yval = qspline(xval)
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

    def testBSpline(self):
        expected = readVector(join(filedir(), 'spline/bSpline.txt'))
        self.evaluateBSpline(expected)

    def testBSplineEqualsBetaSpline(self):
        # only true when beta1=1 and beta2=0
        beta1=1.0
        beta2=0.0
        expected = readVector(join(filedir(), 'spline/bSpline.txt'))
        self.evaluateBetaSpline(beta1, beta2, expected)

    def testBetaSpline(self):
        beta1=1.0
        beta2=0.0
        expected = readVector(join(filedir(), 'spline/betaSpline_1_0.txt'))
        self.evaluateBetaSpline(beta1, beta2, expected)
        print "\n\tbeta1=1.0, beta2=0.0 \tok"

        print "\tbeta1=1.0, beta2=100.0\tok"
        beta1=1.0
        beta2=100.0
        expected = readVector(join(filedir(), 'spline/betaSpline_1_100.txt'))
        self.evaluateBetaSpline(beta1, beta2, expected)

        print "\tbeta1=100.0, beta2=0.0\tok"
        beta1=100.0
        beta2=0.0
        expected = readVector(join(filedir(), 'spline/betaSpline_100_0.txt'))
        self.evaluateBetaSpline(beta1, beta2, expected)

    def testQuadraticSpline(self):
        expected = readVector(join(filedir(), 'spline/qSpline.txt'))
        self.evaluateQuadraticSpline(expected)






suite = allTests(TestSpline)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
