#!/usr/bin/env python

from essentia_test import *
from math import *

class TestMFCC(TestCase):

    def InitMFCC(self, numCoeffs):
        return MFCC(sampleRate = 44100,
                    numberBands = 40,
                    numberCoefficients = numCoeffs,
                    lowFrequencyBound = 0,
                    highFrequencyBound = 11000)


    def testRegression(self):
        # only testing that it yields valid result, but still need to check for
        # correct results.. no ground truth provided
        size = 20
        while (size > 0) :
            bands, mfcc = self.InitMFCC(size)(ones(1024))
            self.assertEqual(len(mfcc), size )
            self.assertEqual(len(bands), 40 )
            self.assert_(not any(numpy.isnan(mfcc)))
            self.assert_(not any(numpy.isinf(mfcc)))
            size -= 1

    def testZero(self):
        # zero input should return dct(lin2db(0)). Try with different sizes
        size = 1024
        val = amp2db(-90.0)
        expected = DCT(inputSize=40, outputSize=13)([val for x in range(40)])
        while (size > 1 ):
            bands, mfcc = MFCC()(zeros(size))
            self.assertEqualVector(mfcc, expected)
            size /= 2

    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(MFCC(), [])
        self.assertComputeFails(MFCC(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(MFCC(), { 'numberBands': 0 })
        self.assertConfigureFails(MFCC(), { 'numberBands': 1 })
        self.assertConfigureFails(MFCC(), { 'numberCoefficients': 0 })
        # number of coefficients cannot be larger than number of bands, otherwise dct
        # will throw and exception
        self.assertConfigureFails(MFCC(), { 'numberBands': 10, 'numberCoefficients':13 })
        self.assertConfigureFails(MFCC(), { 'lowFrequencyBound': -100 })
        # low freq bound cannot be larger than high freq bound
        self.assertConfigureFails(MFCC(), { 'lowFrequencyBound': 100,
                                            'highFrequencyBound': 50 })
        # high freq bound cannot be larger than half the sr
        self.assertConfigureFails(MFCC(), { 'highFrequencyBound': 30000,
                                            'sampleRate': 22050} )

    def testRealCase(self):
        from numpy import mean
        filename = join(testdata.audio_dir, 'recorded','britney.wav')
        audio = MonoLoader(filename=filename, sampleRate=44100)()
        frameGenerator = FrameGenerator(audio, frameSize=1024, hopSize=512)
        window = Windowing(type="blackmanharris62")
        pool=Pool()
        mfccAlgo = self.InitMFCC(13)

        for frame in frameGenerator:
            bands, mfcc = mfccAlgo(window(frame))
            pool.add("bands", bands)
            pool.add("mfcc", mfcc)

        expected = [-8.96260986e+02, 6.09997635e+01, -4.36200752e+01, 2.48095875e+01,
                    -1.55416851e+01, 1.00714369e+01, -6.46236658e+00, 4.16872358e+00,
                    -2.76266885e+00, 1.83512831e+00, -1.36596978e+00, 8.36851299e-01,
                    -4.40462381e-01]


        self.assertAlmostEqualVector(mean(pool['mfcc'], 0), expected, 1.0e-5)


suite = allTests(TestMFCC)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
