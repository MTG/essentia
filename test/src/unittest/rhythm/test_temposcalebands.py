#!/usr/bin/env python

from essentia_test import *


class TestTempoScaleBands(TestCase):

    def testRegression(self):
       # Testing that results are not inf nor nan, but real numbers
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_sweep_0db.wav'),
                           sampleRate = 44100)()

        fft = Spectrum()
        window = Windowing(type = 'hamming')
        nbands = [0, 100, 200, 300, 400, 500, 600, 700, 800]
        fbands = FrequencyBands(frequencyBands = nbands,
                                sampleRate = 44100)
        tempobands = TempoScaleBands()
        bandsgain = [2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5]

        for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
            scaledbands, cumul = tempobands(fbands(fft(window(frame))))

            self.assert_(not any(numpy.isnan(scaledbands)))
            self.assert_(not any(numpy.isinf(scaledbands)))
            self.assert_(all(scaledbands >= 0.0))

            self.assert_(not numpy.isnan(cumul))
            self.assert_(not numpy.isinf(cumul))
            self.assert_(cumul >= 0.0)

    def testConstantInput(self):
       # When input is constant should yiled zero
        bandsgain = [2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5]
        spectrum = [1]*len(bandsgain)
        tempoScale = TempoScaleBands(bandsGain=bandsgain)
        i = 0
        while (i<10):
            scaledbands, cumul = tempoScale(spectrum)
            i+=1

        self.assertEqualVector(scaledbands, zeros(len(bandsgain)))
        self.assertEqual(cumul, 0.0)

    def testZero(self):
        pass
        # Inputting zeros should return null band energies
        fbands = [0, 100, 200, 300, 400, 500, 600, 700, 800]
        scaledbands, cumul = TempoScaleBands()(FrequencyBands(frequencyBands=fbands)(zeros(1024)))
        self.assertEqualVector(scaledbands, zeros(len(fbands)-1))
        self.assertEqual(cumul, 0.0)

    def testInvalidParam(self):
        # Test that we must give valid frequency ranges or order
        self.assertConfigureFails(TempoScaleBands(),{'bandsGain': []})
        self.assertConfigureFails(TempoScaleBands(), { 'frameTime': -1 })
        self.assertComputeFails(TempoScaleBands(bandsGain=[1, 2, 1]), zeros(512))

    def testEmpty(self):
        # Test that FrequencyBands on an empty vector should return null band energies
         self.assertComputeFails(TempoScaleBands(), [])


suite = allTests(TestTempoScaleBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
