#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import AfterMaxToBeforeMaxEnergyRatio as \
        sAfterMaxToBeforeMaxEnergyRatio

class TestAfterMaxToBeforeMaxEnergyRatio_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        strRatio = sAfterMaxToBeforeMaxEnergyRatio()
        p = Pool()

        gen.data >> strRatio.pitch
        strRatio.afterMaxToBeforeMaxEnergyRatio >> (p, 'lowlevel.amtbmer')

        run(gen)

        self.assertRaises(KeyError, lambda: p['lowlevel.amtbmer'])


    def testRegression(self):
        # this algorithm has a standard mode implementation which has been
        # tested thru the unitests in python. Therefore it's only tested that
        # for a certain input standard == streaming
        pitch = readVector(join(filedir(), 'aftermaxtobeforemaxenergyratio', 'input.txt'))

        p = Pool()
        gen = VectorInput(pitch)
        strRatio = sAfterMaxToBeforeMaxEnergyRatio()

        gen.data >> strRatio.pitch
        strRatio.afterMaxToBeforeMaxEnergyRatio >> (p, 'lowlevel.amtbmer')

        run(gen)

        stdResult = AfterMaxToBeforeMaxEnergyRatio()(pitch)
        strResult = p['lowlevel.amtbmer']
        self.assertAlmostEqual(strResult, stdResult, 5e-7)


suite = allTests(TestAfterMaxToBeforeMaxEnergyRatio_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
