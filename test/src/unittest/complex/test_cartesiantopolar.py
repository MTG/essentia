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
from math import *

class TestCartesianToPolar(TestCase):

    def testZero(self):
        inputc = numpy.array([ complex() ] * 4, dtype='c8')

        mag, phase = CartesianToPolar()(inputc)
        self.assertEqualVector(mag, zeros(4))
        self.assertEqualVector(phase, zeros(4))

    def testRegression(self):
        inputc = [ (1, -5), (2, -6), (-3, 7), (-4, 8) ]
        inputc = numpy.array([ complex(*c) for c in inputc ], dtype='c8')

        expectedMag = [ 5.0990, 6.3246, 7.6158, 8.9443 ]
        expectedPhase = [ -1.3734, -1.2490, 1.9757, 2.0344 ]

        c2p = CartesianToPolar()
        self.assertAlmostEqualVector(c2p(inputc)[0], expectedMag, 1e-4)
        self.assertAlmostEqualVector(c2p(inputc)[1], expectedPhase, 1e-4)

    def testCircle(self):
        # Tests a few points, including one at phase=pi, separately and then all together at the same time
        c2p = CartesianToPolar()

        circle = { (1, 0): (1, 0),
                   (sqrt(2)/2, sqrt(2)/2): (1, pi/4),
                   (1, 1): (sqrt(2), pi/4),
                   (0, 1): (1, pi/2),
                   (-1, 0): (1, pi),
                   (0, -1): (1, -pi/2)
                   }

        for c, p in circle.items():
            mag, phase = c2p(numpy.array([complex(*c)], dtype='c8'))

            self.assertAlmostEqualVector(mag, [ p[0] ])
            self.assertAlmostEqualVector(phase, [ p[1] ])

        circleCart = circle.keys()
        circlePolar = [ circle[key] for key in circleCart ]

        circleCart = numpy.array([ complex(*c) for c in circleCart ], dtype='c8')
        circleMag = [ c[0] for c in circlePolar ]
        circlePhase = [ c[1] for c in circlePolar ]

        cmag, cphase = c2p(circleCart)

        self.assertAlmostEqualVector(cmag, circleMag, 1e-6)
        self.assertAlmostEqualVector(cphase, circlePhase, 1e-6)


suite = allTests(TestCartesianToPolar)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
