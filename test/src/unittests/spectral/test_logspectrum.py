#!/usr/bin/env python

# Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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

class TestLogSpectrum(TestCase):

    def testRegression(self, frameSize=8192 + 1):
        expected = array([  29.521513 ,   27.441898 ,   26.655254 ,   26.23174  ,
                            25.356936 ,   24.296692 ,   23.467793 ,   23.073942 ,
                            22.010378 ,   19.826815 ,   17.447205 ,   15.999705 ,
                            15.499856 ,   14.483567 ,   13.10268  ,   12.109021 ,
                            11.780534 ,   11.255134 ,   10.505994 ,   10.007475 ,
                            10.294797 ,   11.649873 ,   13.241921 ,   14.04129  ,
                            15.0936365,   17.07475  ,   18.479214 ,   18.365091 ,
                            17.354443 ,   16.441502 ,   15.803897 ,   14.0882225,
                            12.459978 ,   11.872289 ,   10.973168 ,   10.272282 ,
                            10.93928  ,   12.780078 ,   13.75453  ,   13.902253 ,
                            14.03201  ,   14.007873 ,   13.837217 ,   13.694722 ,
                            13.274383 ,   12.808523 ,   11.746238 ,   10.068007 ,
                            9.801479 ,   10.025723 ,   10.36356  ,   11.180739 ,
                            11.024283 ,    9.669961 ,    9.211928 ,    9.143377 ,
                            9.324788 ,    9.722918 ,    9.9669485,   10.06157  ,
                            8.945379 ,    8.352387 ,    8.667179 ,    8.645229 ,
                            8.466478 ,    8.145741 ,    8.102758 ,    8.339408 ,
                            9.091846 ,   10.907492 ,   12.577294 ,    9.844134 ,
                            8.88199  ,    8.731721 ,    8.564365 ,    7.089214 ,
                            6.820865 ,    7.2314105,    7.824168 ,    8.3441   ,
                            9.000613 ,    9.578227 ,    9.352688 ,   11.576548 ,
                            17.15378  ,   21.598444 ,   23.76267  ,   30.632256 ,
                            49.047886 ,   77.0526   ,  102.86156  ,  122.54504  ,
                            137.17317  ,  160.752    ,  261.5374   ,  389.9752   ,
                            609.9252   ,  877.13715  ,  679.7323   ,  379.14688  ,
                            321.35117  ,  375.1989   ,  418.306    ,  534.0889   ,
                            509.3253   ,  666.0269   , 1242.0575   , 2838.5713   ,
                            3155.8198   , 1829.4911   ,  940.0592   ,  470.50735  ,
                            418.63632  ,  543.2316   ,  722.0029   ,  656.54913  ,
                            787.5359   ,  555.84247  ,  377.14426  ,  277.57123  ,
                            259.91522  ,  244.62334  ,  189.55319  ,  149.07292  ,
                            129.34392  ,  165.01091  ,  195.81532  ,  136.49275  ,
                            76.68514  ,   97.11707  ,  151.90732  ,  224.73782  ,
                            337.18192  ,  332.75058  ,  276.9806   ,  369.5457   ,
                            423.7845   ,  385.96686  ,  396.63702  ,  417.4205   ,
                            468.57962  ,  531.24524  ,  657.8269   , 1281.0719   ,
                            1458.027    ,  698.2347   ,  314.64862  ,  262.41177  ,
                            250.06857  ,  436.16797  ,  742.8968   ,  696.05994  ,
                            698.1335   ,  749.1297   ,  626.77167  ,  716.98303  ,
                            650.1398   , 1007.5442   , 1078.1001   ,  924.02277  ,
                            954.38226  ,  983.3749   , 1020.8519   , 1221.412    ,
                            2240.7656   , 2727.331    , 1418.2369   ,  885.09     ,
                            694.7146   ,  428.4301   ,  396.5824   ,  593.49005  ,
                            566.84216  ,  518.056    ,  460.924    ,  332.4222   ,
                            290.82968  ,  236.83658  ,  266.93393  ,  353.3882   ,
                            270.42496  ,  107.788536 ,   84.7718   ,   91.697174 ,
                            80.34272  ,   69.842865 ,  130.54848  ,  125.181015 ,
                            129.66034  ,  138.58502  ,  107.673805 ,  175.68202  ,
                            182.88759  ,  137.9704   ,   92.93435  ,   71.129166 ,
                            78.80348  ,  114.65561  ,  125.607994 ,  125.94448  ,
                            241.2471   ,  270.2437   ,  169.07764  ,  151.46762  ,
                            174.52895  ,  150.08052  ,  143.80923  ,  181.24687  ,
                            262.3386   ,  391.69852  ,  237.67671  ,  149.62773  ,
                            137.92833  ,  178.46875  ,  257.16058  ,  616.3291   ,
                            660.309    ,  402.2011   ,  349.27042  ,  272.2735   ,
                            245.1808   ,  255.92816  ,  241.11888  ,  142.76396  ,
                            95.74622  ,   79.352    ,   71.41603  ,  109.63512  ,
                            69.190254 ,   38.56327  ,   29.309353 ,   31.107338 ,
                            44.70423  ,   35.44568  ,   22.666018 ,   16.979908 ,
                            32.010136 ,   45.12984  ,   34.377556 ,   40.05504  ,
                            79.10235  ,  127.51099  ,  110.30661  ,   95.92048  ,
                            86.051285 ,   97.34698  ,   89.825485 ,   62.368706 ,
                            67.440575 ,   55.217617 ,   40.645    ,   69.08143  ,
                            58.60874  ,   40.766853 ,   52.087135 ,   75.063484 ])

        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded/vignesh.wav'),
                    sampleRate = 44100)()

        w = Windowing(type='hann', normalized=False)
        spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
        logspectrum = LogSpectrum(frameSize=frameSize)

        logfreqspectrogram = []
        for frame in FrameGenerator(audio, frameSize=16384, hopSize=2048,
                                    startFromZero=True):
            logfreqspectrum, _, _ = logspectrum(spectrum(w(frame)))
            logfreqspectrogram.append(logfreqspectrum)
        logfreqspectrogram = array(logfreqspectrogram).mean(axis=0)

        self.assertAlmostEqualVector(logfreqspectrogram, expected, 1e-4)

    def testZero(self):
        # Inputting zeros should return zero. Try with different sizes
        size = 1024
        while (size >= 256 ):
            self.assertEqualVector(LogSpectrum(frameSize = size)(zeros(size))[0], zeros(256))
            size = int(size/2)

    def testInvalidInput(self):
        self.assertComputeFails(LogSpectrum(), [])
        self.assertComputeFails(LogSpectrum(), [0.5])


    def testInvalidParam(self):
        self.assertConfigureFails(LogSpectrum(), { 'frameSize': 1 })
        self.assertConfigureFails(LogSpectrum(), { 'sampleRate': 0 })
        self.assertConfigureFails(LogSpectrum(), { 'rollOn': -1})

    def testWrongInputSize(self):
        # This test makes sure that even though the frameSize given at
        # configure time does not match the input spectrum, the algorithm does
        # not crash and correctly resizes internal structures to avoid errors.
        print('\n')
        self.testRegression(frameSize=1000)
        print('...')


suite = allTests(TestLogSpectrum)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
