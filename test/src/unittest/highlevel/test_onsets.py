#!/usr/bin/env python

from essentia_test import *


class TestOnsets(TestCase):

    def testZero(self):
        # zeros should return no onsets (empty array)
        n = 10
        detection = zeros(100).reshape(n,n)
        weights = ones(n)
        self.assertEqualMatrix(Onsets()(detection, weights), [])

    def testConstantInput(self):
        # constant detection function should return first position:
        n = 10
        detection = ones(100).reshape(n,n)
        weights = ones(n)
        size = 2048
        sr = 44100.0
        while (size > 32):
            hopsize = size/2
            frameTime = (size - hopsize)/sr
            frameRate = 1.0/frameTime
            if (size == 1024 and hopsize == 512):
                onset = Onsets(frameRate=frameRate)(detection, weights)
                self.assertAlmostEqualVector(onset, array([frameTime]))
            else:
                # Onsets does not support other framerates than
                # (1024-512)/44100
                self.assertConfigureFails(Onsets(), { 'frameRate': frameRate })
            size /= 2

    def testImpulse(self):
        # Given an impulse should return its position
        n = 10
        detection = zeros(100).reshape(n,n)
        for i in range(len(detection)):
            detection[i, 5] = 1
            detection[i, 4] = .2
            detection[i, 6] = .3
        weights = ones(n)
        size = 2048
        sr = 44100.0
        while (size > 32):
            hopsize = size/2
            frameTime = (size - hopsize)/sr
            frameRate = 1.0/frameTime
            if (size == 1024 and hopsize == 512):
                onset = Onsets(frameRate=frameRate)(detection, weights)
                self.assertAlmostEqualVector( onset, array([4*frameTime]), 1e-5)
            else:
                # Onsets does not support other framerates than
                # (1024-512)/44100
                self.assertConfigureFails(Onsets(), { 'frameRate': frameRate })
            size /= 2

    def testInvalidParam(self):
        self.assertConfigureFails(Onsets(), { 'frameRate':-1 })
        self.assertConfigureFails(Onsets(), { 'alpha': 2 })
        self.assertConfigureFails(Onsets(), { 'delay': -1 })
        self.assertConfigureFails(Onsets(), { 'silenceThreshold':10 })

    def testEmpty(self):
        # Empty input should raise an exception
        self.assertComputeFails(Onsets(), array([[]]), [])


    def testBadWeightSize(self):
        weights = [1,2,3,4]
        input = [[1,2,3,4],
                 [5,6,7,8],
                 [9,10,11,12]]

        self.assertComputeFails(Onsets(), input, weights)


suite = allTests(TestOnsets)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
