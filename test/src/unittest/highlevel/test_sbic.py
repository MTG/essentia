#!/usr/bin/env python

from numpy import array
from essentia_test import *

class TestSBic(TestCase):

    def atestNotEnoughFrames(self):
        self.assertComputeFails( SBic(), array([[]]) )
        self.assertComputeFails( SBic(), array([[1]]) )
        self.assertComputeFails( SBic(), array([[1], [1], [1]]) )

    def atestOneFeature(self):
        features = array([[0, 1, 2, 3, 4]])
        self.assertEqualVector(SBic(minLength=1)(features), [0, len(features[0])-1])

    # the following test is commented, as it fails due to rounding errors.
    # to solve this problem a quick solution would be to use a threshold higher
    # than 1e-10 (i.e. log10 = diag_cov > 1e-6 ? logd : -6
    def atestConstantFeature(self):
        features = array([ [10]*1000 ])
        self.assertEqualVector(SBic()(features), [0, len(features[0])-1])

    def atestTwoSegments(self):

        # Half 1s and half 0s
        # [ [ 1, ..., 1, 0, ..., 0],
        #   [ 1, ..., 1, 0, ..., 0] ]
        features = array( [ [1]*200 + [0]*200 ] +
                          [ [1]*200 + [0]*200 ])

        segments = SBic()(features)

        self.assertAlmostEqualVector(segments, [0, 199, 399], .2)


    # The following test is commented because for some reason reducing the
    # increment parameters create a lot of false positives (incorrect
    # segmentation points). This is probably due to the fact that the BIC is
    # trying to overfit the given data.

    def atestSmallIncs(self):
        # Half 1s and half 0s
        # [ [ 1, ..., 1, 0, ..., 0],
        #   [ 1, ..., 1, 0, ..., 0] ]
        # This test causes duplicates in the segmentation array, and these
        # duplicates caused a crash due to empty subarrays being created
        # (because from one segment to the next is zero length, because they
        # are the same position (sorry if that didn't make any sense)).
        features = array( [ [1]*200 + [0]*200 ] +
                          [ [1]*200 + [0]*200 ])

        segments = SBic(inc1=4, inc2=2)(features)

        self.assertAlmostEqualVector(segments, [0, 199, 399], .1)


    def atestSmallMinLength(self):
        features = array( [ [1]*200 + [0]*200 ] +
                          [ [1]*200 + [0]*200 ])

        segments = SBic(minLength=1)(features)
        self.assertAlmostEqualVector(segments, [0, 199, 399], .2)


    def atestLargeMinLength(self):
        loader = MonoLoader(filename = join(testdata.audio_dir, 'recorded',
                                            '01-Allegro__Gloria_in_excelsis_Deo_in_D_Major.wav'),
                            downmix='left', sampleRate=441000)

        if sys.platform == 'win32' and getattr(loader, 'hasDoubleCompute', False):
            print 'WARNING: skipping this test as Windows seems to do weird things with memory...'
            return

        audio = loader()

        w = Windowing(type='blackmanharris62', size=2048)
        s = Spectrum(size=2048)
        m = MFCC(highFrequencyBound=8000)
        features = []

        for frame in FrameGenerator(audio, frameSize=2048, hopSize=1024):
            if isSilent(frame):
                continue

            (_,mfcc) = m(s(w(frame)))
            features.append(mfcc)

        # compute transpose of features array
        features_transpose = []
        for i in range(len(features[0])):
            featureVals = []
            for f in features:
                featureVals.append(f[i])
            features_transpose.append(featureVals)

        features_transpose = array(features_transpose)
        nFrames = len(features)
        segments = SBic(minLength=nFrames*2, cpw=1.5, size1=1000,
                        inc1=300, size2=600, inc2=50)(features_transpose)

        # since the minLength is so high, the entire audio signal should be
        # considered as one segment
        expected = [0, nFrames-1]
        self.assertEqualVector(segments, expected)


    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded',\
                           '01-Allegro__Gloria_in_excelsis_Deo_in_D_Major.wav'),
                           downmix='left', sampleRate=44100)()

        w = Windowing(type='blackmanharris62', size=2048)
        s = Spectrum(size=2048)
        m = MFCC(highFrequencyBound=8000)
        features = []

        for frame in FrameGenerator(audio, frameSize=2048, hopSize=1024):
            (_,mfcc) = m(s(w(frame)))
            features.append(mfcc)

        # compute transpose of features array
        features_transpose = []
        for i in range(len(features[0])):
            featureVals = []
            for f in features:
                featureVals.append(f[i])
            features_transpose.append(featureVals)

        features_transpose = array(features_transpose)
        segments = SBic(cpw=1.5, size1=1000, inc1=300, size2=600, inc2=50)(features_transpose)
        expected = [0., 598., 1097., 1796., 2445., 2894., 3693., 4442., 5641., 5728.]
        self.assertEqualVector(segments, expected)

    def atestMinLengthEqualToAudioFrames(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded',\
                           'britney.wav'),
                           downmix='left', sampleRate=441000)()

        w = Windowing(type='blackmanharris62', size=2048)
        s = Spectrum(size=2048)
        m = MFCC(highFrequencyBound=8000)
        features = []

        for frame in FrameGenerator(audio, frameSize=2048, hopSize=1024):
            if isSilent(frame):
                continue

            (_,mfcc) = m(s(w(frame)))
            features.append(mfcc)

        # compute transpose of features array
        features_transpose = []
        for i in range(len(features[0])):
            featureVals = []
            for f in features:
                featureVals.append(f[i])
            features_transpose.append(featureVals)

        bands, nFrames = numpy.shape(features_transpose)
        features_transpose = array(features_transpose)
        sbic = SBic(cpw=1.5, size1=1000, inc1=300, size2=600, inc2=50, minLength=nFrames)
        segments = sbic(features_transpose)

        expected = [0., nFrames-1]
        self.assertEqualVector(segments, expected)

    def atestMinLengthLargerThanAudioFrames(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded',\
                           'britney.wav'),
                           downmix='left', sampleRate=441000)()

        w = Windowing(type='blackmanharris62', size=2048)
        s = Spectrum(size=2048)
        m = MFCC(highFrequencyBound=8000)
        features = []

        for frame in FrameGenerator(audio, frameSize=2048, hopSize=1024):
            if isSilent(frame):
                continue

            (_,mfcc) = m(s(w(frame)))
            features.append(mfcc)

        # compute transpose of features array
        features_transpose = []
        for i in range(len(features[0])):
            featureVals = []
            for f in features:
                featureVals.append(f[i])
            features_transpose.append(featureVals)

        bands, nFrames = numpy.shape(features_transpose)
        features_transpose = array(features_transpose)
        sbic = SBic(cpw=1.5, size1=1000, inc1=300, size2=600, inc2=50, minLength=nFrames+2)
        segments = sbic(features_transpose)

        expected = [0., nFrames-1]
        self.assertEqualVector(segments, expected)

    def atestSize2LargerThanSize1(self):
        # Half 1s and half 0s
        # [ [ 1, ..., 1, 0, ..., 0],
        #   [ 1, ..., 1, 0, ..., 0] ]
        from numpy.random import normal
        features = zeros([2, 400])
        for i in range(200):
            features[0][i] = normal()
            features[1][i] = normal()

        segments = SBic(size1=25, size2=50)(features)

        self.assertAlmostEqualVector(segments, [0, 199, 399], .15)

suite = allTests(TestSBic)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
