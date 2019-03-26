#!/usr/bin/env python

# Copyright (C) 2006-2017  Music Technology Group - Universitat Pompeu Fabra
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
from essentia import array, run, Pool
import essentia.standard as estd
import essentia.streaming as ess
from essentia_test import *
import numpy as np


def hpcp(audio_vector):
    """
    Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode 
    For full list of parameters of essentia standard mode HPCP please refer to http://essentia.upf.edu/documentation/reference/std_HPCP.html
    """
    audio = array(audio_vector)
    frameGenerator = estd.FrameGenerator(audio, frameSize=4096, hopSize=2048)
    window = estd.Windowing(type='blackmanharris62')
    spectrum = estd.Spectrum()
    spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=1e-05,
                                        maxFrequency=3500,
                                        minFrequency=100,
                                        maxPeaks=100,
                                        orderBy="frequency",
                                        sampleRate=44100)
    spectralWhitening = estd.SpectralWhitening(maxFrequency= 3500,
                                                sampleRate=44100)
    hpcp = estd.HPCP(sampleRate=44100,
                    maxFrequency=3500,
                    minFrequency=100,
                    referenceFrequency=440,
                    nonLinear=True,
                    harmonics=8,
                    size=12)
    #compute hpcp for each frame and add the results to the pool
    for frame in frameGenerator:
        spectrum_mag = spectrum(window(frame))
        frequencies, magnitudes = spectralPeaks(spectrum_mag)
        w_magnitudes = spectralWhitening(spectrum_mag,
                                        frequencies,
                                        magnitudes)
        hpcp_vector = hpcp(frequencies, w_magnitudes)
    return hpcp_vector


def cross_recurrent_plot(input_x, input_y, tau=1, m=1, kappa=0.095):
    """
    Constructs the Cross Recurrent Plot of two audio feature vector
    """
    from sklearn.metrics.pairwise import euclidean_distances
    pdistances = euclidean_distances(input_x, input_y)
    #pdistances = resample_simmatrix(pdistances)
    transposed_pdistances = pdistances.T
    eph_x = np.percentile(pdistances, kappa*100, axis=1)
    eph_y = np.percentile(transposed_pdistances, kappa*100, axis=1)
    x = eph_x[:,None] - pdistances
    y = eph_y[:,None] - transposed_pdistances
    #apply heaviside function to the array (Binarize the array)
    x = np.piecewise(x, [x<0, x>=0], [0,1])
    y = np.piecewise(y, [y<0, y>=0], [0,1])
    crp = x*y.T
    return crp


class TestChromaCrossSimilarity(TestCase):

    query = estd.MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'mozart_c_major_30sec.wav'))()
    reference = estd.MonoLoader(filename=join(testdata.audio_dir, 'recorded', 'Vivaldi_Sonata_5_II_Allegro.wav'))
    query_hpcp = hpcp(query)
    reference_hpcp = hpcp(reference)
    expected = cross_recurrent_plot(query_hpcp, reference_hpcp)

    def testEmpty(self):
        self.assertComputeFails(estd.CrossSimilarityMatrix(), [])

    def testRegressionStandard(self):
        """Tests standard ChromaCrossSimilarity algo"""
        csm = estd.ChromaCrossSimilarity(oti=False, stackFrameSize=1)
        result = csm.compute(self.query_hpcp, self.reference_hpcp)
        self.assertAlmostEqual(np.mean(self.expected), np.mean(result))
        self.assertAlmostEqualVector(self.expected, result)

    def testInvalidParam(self):
        self.assertConfigureFails(estd.ChromaCrossSimilarity(), { 'binarizePercentile': -1 })
        self.assertConfigureFails(estd.ChromaCrossSimilarity(), { 'otiBinary': -1 })

    def testOTIBinaryCompute(self):
        """Tests standard ChromaCrossSimilarity algo when param 'otiBinary=True'"""
        # test oti-based binary sim matirx method
        self.assertComputeFails(estd.ChromaCrossSimilarity(otiBinary=True), [])

    def testRegressionStreaming(self):
        """Tests streaming ChromaCrossSimilarity algo against the standard mode algorithm with 'streamingMode=True' """
        # compute chromacrosssimilarity matrix using standard mode
        csm_standard = estd.ChromaCrossSimilarity(streamingMode=True)
        sim_matrix_std = csm_standard.compute(self.query_hpcp, self.reference_hpcp)
        # compute chromacrosssimilarity matrix using streaming mode
        queryVec = ess.VectorInput(self.query_hpcp)
        csm_streaming = ess.ChromaCrossSimilarity(referenceFeature=self.reference_hpcp, oti=0)
        pool = Pool()
        queryVec.data >> csm_streaming.queryFeature
        csm_streaming.csm >>  (pool, 'csm')

        self.assertAlmostEqualVector(sim_matrix_std, pool['csm'])
        self.assertAlmostEqual(np.mean(sim_matrix_std), np.mean(pool['csm']))
    

suite = allTests(TestCrossSimilarityMatrix)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)

