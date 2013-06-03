#!/usr/bin/env python

# Copyright (C) 2006-2008 Music Technology Group (MTG)
#                         Universitat Pompeu Fabra


from essentia_test import *
from math import *

class TestMagnitude(TestCase):

    def testZero(self):
        inputc = numpy.array([ complex() ] * 4, dtype='c8')
        
        self.assertEqualVector(Magnitude()(inputc), zeros(4))

    def testEmpty(self):
        self.assertEqualVector(Magnitude()(numpy.array([],dtype='c8')), [])

    def testRegression(self):
        inputc = [ (1, -5), (2, -6), (-3, 7), (-4, 8) ]
        inputc = numpy.array([ complex(*c) for c in inputc ], dtype='c8')
        expected = array([ abs(c) for c in inputc ]) 

        self.assertAlmostEqualVector(Magnitude()(inputc), expected)


suite = allTests(TestMagnitude)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

