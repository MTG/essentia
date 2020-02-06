#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
import numpy as np

class TestMelBands(TestCase):

    def InitMelBands(self, sr, nbands, scale, normalize):
        return MelBands(inputSize=1025,
                        sampleRate=sr,
                        numberBands=nbands,
                        lowFrequencyBound=0,
                        highFrequencyBound=sr/2,
                        warpingFormula=scale,
                        type='magnitude',
                        weighting='linear',
                        normalize=normalize)

    def testFlatSpectrum(self):
        spectrum = [1]*1025
        mbands = self.InitMelBands(44100, 24, 'htkMel', 'unit_sum')(spectrum)
        self.assertEqual(len(mbands), 24 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*24, 1e-5)

        mbands = self.InitMelBands(44100, 128, 'htkMel', 'unit_sum')(spectrum)
        self.assertEqual(len(mbands), 128 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*128, 1e-5)

        mbands = self.InitMelBands(44100, 24, 'slaneyMel', 'unit_sum')(spectrum)
        self.assertEqual(len(mbands), 24 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*24, 1e-5)

        mbands = self.InitMelBands(44100, 128, 'slaneyMel', 'unit_sum')(spectrum)
        self.assertEqual(len(mbands), 128 )
        self.assert_(not any(numpy.isnan(mbands)))
        self.assert_(not any(numpy.isinf(mbands)))
        self.assertAlmostEqualVector(mbands, [1]*128, 1e-5)


    def testZeroSpectrum(self):
        # Inputting zeros should return zero. Try with different sizes.
        size = 1024
        while (size >= 256 ):
            self.assertEqualVector(MelBands()(zeros(size)), zeros(24))
            size = size // 2


    def testRegression(self):
        # Compare to another reference implementation (Librosa).
        spectrum = [0.1 * i for i in range(8)] * 128 + [0.5]

        # Test with 'unit_max' normalization.
        mbands = self.InitMelBands(44100, 24, 'htkMel', 'unit_max')(spectrum)
        expected = [ 2.05486465,  2.0094972 ,  2.38035517,  2.79824033,  3.16617682,
                     3.61716034,  4.32444713,  4.80223426,  5.54741135,  6.36711453,
                     7.40140018,  8.43368144,  9.73281032, 11.14577405, 12.84719231,
                    14.76528787, 16.92703011, 19.51814534, 22.37584746, 25.76257315,
                    29.59672515, 34.01594688, 39.10000643, 44.9481521 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(22050, 24, 'htkMel', 'unit_max')(spectrum)
        expected = [ 2.91459531,  3.19272426,  3.5437988 ,  4.16714568,  4.39816764,
                     5.10022135,  5.68411513,  6.3433626 ,  7.05882067,  7.94480316,
                     8.89159678,  9.94189181, 11.12717903, 12.45546612, 13.95883964,
                    15.60566105, 17.45834884, 19.58830233, 21.87743455, 24.50283962,
                    27.44133832, 30.69866997, 34.37137694, 38.47261776 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(16000, 24, 'htkMel', 'unit_max')(spectrum)
        expected = [ 3.42785724,  3.95063344,  4.23818102,  4.79472915,  5.21478985,
                     5.77134851,  6.42717578,  7.10003072,  7.84686725,  8.66048351,
                     9.63173673, 10.5930362 , 11.71974037, 13.00698499, 14.3715635 ,
                    15.87041045, 17.56843056, 19.43649752, 21.49850978, 23.77058851,
                    26.29715546, 29.08617578, 32.16698756, 35.58753042 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(44100, 24, 'slaneyMel', 'unit_max')(spectrum)
        expected = [ 2.59826712,  2.62543634,  2.60776263,  2.60826131,  2.61345994,
                     2.67922738,  3.02142148,  3.69658401,  4.09855892,  5.06282636,
                     5.93030468,  6.89536808,  8.23276783,  9.67149053, 11.42328897,
                    13.44593751, 15.86146257, 18.71179058, 22.09087055, 26.02358676,
                    30.70081553, 36.21068794, 42.70093326, 50.3710549 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(22050, 24, 'slaneyMel', 'unit_max')(spectrum)
        expected = [ 4.41106576,  4.21484495,  4.45460042,  4.19350371,  4.45513146,
                     4.2138322 ,  4.48001245,  4.77897885,  5.46124375,  6.27005486,
                     7.27131606,  8.26265848,  9.53799884, 10.8703333 , 12.52660421,
                    14.31964489, 16.48658923, 18.87400565, 21.64632984, 24.84907417,
                    28.4961605 , 32.70193018, 37.48933891, 43.02779959 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(16000, 24, 'slaneyMel', 'unit_max')(spectrum)
        expected = [ 5.40342656,  5.41845359,  5.40706262,  5.41121639,  5.40924425,
                     5.40543361,  5.40997146,  5.53830564,  6.08920434,  6.94842373,
                     7.81450219,  8.84777817, 10.07150674, 11.36904891, 12.87900428,
                    14.61011763, 16.4970834 , 18.72118075, 21.19401982, 24.00157627,
                    27.18831724, 30.76454194, 34.87460463, 39.47271554 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        # Test with 'unit_tri' normalization.
        mbands = self.InitMelBands(44100, 24, 'htkMel', 'unit_tri')(spectrum)
        expected = [ 0.01828156, 0.015554  , 0.01602954, 0.01639416, 0.01613853,
                     0.01604062, 0.01668432, 0.01611929, 0.0162001 , 0.01617688,
                     0.01636027, 0.01621879, 0.01628412, 0.01622411, 0.01626985,
                     0.01626829, 0.01622578, 0.01627751, 0.01623505, 0.01626252,
                     0.01625425, 0.01625289, 0.0162536 , 0.01625583 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(22050, 24, 'htkMel', 'unit_tri')(spectrum)
        expected = [ 0.03292591, 0.03222258, 0.03195268, 0.0335673 , 0.0316511 ,
                     0.03279031, 0.03264817, 0.03255029, 0.03235988, 0.03253848,
                     0.03253369, 0.0324984 , 0.03249507, 0.03249614, 0.03253573,
                     0.03249621, 0.03247829, 0.03255564, 0.03248368, 0.03250309,
                     0.03252018, 0.03250174, 0.03251048, 0.03251005 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(16000, 24, 'htkMel', 'unit_tri')(spectrum)
        expected = [ 0.04384827, 0.04568984, 0.04431549, 0.04532766, 0.04457169,
                     0.04459876, 0.04490439, 0.04484892, 0.04481375, 0.04471778,
                     0.04496409, 0.04471002, 0.04472244, 0.0448753 , 0.04482888,
                     0.04475743, 0.04479537, 0.04480653, 0.04480792, 0.04479294,
                     0.04480243, 0.04480253, 0.04479705, 0.04480846 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(44100, 24, 'slaneyMel', 'unit_tri')(spectrum)
        expected = [ 0.01624121, 0.01641104, 0.01630056, 0.01630368, 0.01633617,
                     0.01615584, 0.01612855, 0.01671312, 0.01571219, 0.01645687,
                     0.0163448 , 0.0161142 , 0.01631344, 0.01624956, 0.01627376,
                     0.01624189, 0.01624565, 0.01625017, 0.01626686, 0.01624825,
                     0.01625317, 0.01625449, 0.01625259, 0.01625604 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(22050, 24, 'slaneyMel', 'unit_tri')(spectrum)
        expected = [ 0.03314226, 0.03166796, 0.03346935, 0.03150762, 0.03347334,
                     0.03166035, 0.03312444, 0.0324344 , 0.0323428 , 0.03237033,
                     0.03272491, 0.03241717, 0.03262137, 0.03240989, 0.03255801,
                     0.0324449 , 0.03256378, 0.03249809, 0.03249135, 0.03251497,
                     0.03250493, 0.03251817, 0.03249752, 0.03251482 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)

        mbands = self.InitMelBands(16000, 24, 'slaneyMel', 'unit_tri')(spectrum)
        expected = [ 0.04478409, 0.04490864, 0.04481423, 0.04484866, 0.04483231,
                     0.04480073, 0.04483834, 0.04465683, 0.04474766, 0.04501282,
                     0.04470051, 0.04468963, 0.04491881, 0.04477331, 0.04478563,
                     0.04486125, 0.04472868, 0.04482017, 0.0448038 , 0.04480253,
                     0.04481325, 0.04477503, 0.04481836, 0.04479247 ]
        self.assertAlmostEqualVector(mbands, expected, 1e-5)


    def testRegressionHtkMode(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                           sampleRate = 44100)()*2**15
        expected = [ 10.35452019,  12.97260263,  13.87114479,  12.92819811,  13.53927989,
                     13.65001411,  13.7067006,   12.72165126,  12.16052112,  12.29371287,
                     12.49577573,  12.68672873,  13.08112941,  12.59404232,  11.71325616,
                     11.48389955,  12.27751253,  12.07873884,  12.02260756,  12.42848721,
                     11.13694966,  10.49976274,  11.3370437,   12.06821492,  12.1631667,
                     11.84755549]

        frameSize = 1102
        hopSize = 441
        fftsize = 2048
        paddingSize = fftsize - frameSize
        spectrumSize = int(fftsize/2) + 1
        w = Windowing(type = 'hamming', 
                      size = frameSize, 
                      zeroPadding = paddingSize,
                      normalized = False,
                      zeroPhase = False)

        spectrum = Spectrum(size = fftsize)

        mbands = MelBands(inputSize= spectrumSize,
                          type = 'magnitude',
                          highFrequencyBound = 8000,
                          lowFrequencyBound = 0,
                          numberBands = 26,
                          warpingFormula = 'htkMel',
                          weighting = 'linear', 
                          normalize = 'unit_max')

        pool = Pool()
        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize, startFromZero = True, validFrameThresholdRatio = 1):
            pool.add('melBands', mbands(spectrum(w(frame))))

        self.assertAlmostEqualVector(np.mean(np.log(pool['melBands']),0), expected, 1e-2)


    def testInvalidInput(self):
        # mel bands should fail for a spectrum with less than 2 bins
        self.assertComputeFails(MelBands(), [])
        self.assertComputeFails(MelBands(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(MelBands(), { 'numberBands': 0 })
        self.assertConfigureFails(MelBands(), { 'numberBands': 1 })
        self.assertConfigureFails(MelBands(), { 'lowFrequencyBound': -100 })
        self.assertConfigureFails(MelBands(), { 'lowFrequencyBound': 100,
                                                'highFrequencyBound': 50 })
        self.assertConfigureFails(MelBands(), { 'highFrequencyBound': 30000,
                                                'sampleRate': 22050})

    def testWrongInputSize(self):
        # This test makes sure that even though the inputSize given at
        # configure time does not match the input spectrum, the algorithm does
        # not crash and correctly resizes internal structures to avoid errors.
        spec = [.1,.4,.5,.2,.1,.01,.04]*100
        self.assertAlmostEqualVector(
                MelBands(inputSize=1024, sampleRate=10, highFrequencyBound=4)(spec),
                MelBands(inputSize=len(spec), sampleRate=10, highFrequencyBound=4)(spec),
                1e-6)


    def testNotEnoughSpectrumBins(self):
        self.assertConfigureFails(MelBands(), {'numberBands': 256, 
                                               'inputSize': 1025})


suite = allTests(TestMelBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
