#!/usr/bin/env python

from essentia_test import *

class TestEnergy(TestCase):

    def testZero(self):
        self.assertEqual(Energy()(zeros(1000)), 0)

    def testEmptyOrOne(self):
        self.assertComputeFails(Energy(), [])
        
        self.assertEqual(Energy()([23]), 23*23)


    def testRegression(self):
        inputArray = readVector(join(filedir(), 'stats/input.txt'))
        basicDesc = readVector(join(filedir(), 'stats/basicdesc.txt'))

        self.assertAlmostEqual(Energy()(inputArray), basicDesc[6])



suite = allTests(TestEnergy)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

