#!/usr/bin/env python
from essentia_test import *
from essentia.streaming import FlatnessSFX as sFlatnessSFX

class TestFlatnessSfx_Streaming(TestCase):

    def testRegression(self):
        # this algorithm has a standard mode implementation which has been
        # tested thru the unitests in python. Therefore it's only tested that
        # for a certain input standard == streaming
        envelope = range(22050)
        envelope.reverse()
        envelope = range(22050) + envelope

        # Calculate standard result
        stdResult = FlatnessSFX()(envelope)

        # Calculate streaming result
        p = Pool()
        input = VectorInput(envelope)
        accu = RealAccumulator()
        strFlatnessSfx = sFlatnessSFX()

        input.data >> accu.data
        accu.array >> strFlatnessSfx.envelope
        strFlatnessSfx.flatness >> (p, 'lowlevel.flatness')

        run(input)
        strResult = p['lowlevel.flatness']

        # compare results
        self.assertEqual(len(strResult), 1)
        self.assertAlmostEqual(strResult[0], stdResult, 5e-7)


suite = allTests(TestFlatnessSfx_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
