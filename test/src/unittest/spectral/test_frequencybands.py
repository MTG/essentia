#!/usr/bin/env python

from essentia_test import *


class TestFrequencyBands(TestCase):

    def testRegression(self):
        # Simple regression test, comparing normal behaviour
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_sweep_0db.wav'),
                           sampleRate = 44100)()

        fft = Spectrum()
        window = Windowing(type = 'hamming')
        fbands = FrequencyBands(sampleRate = 44100)

        for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
            bands = fbands(fft(window(frame)))

            self.assert_(not any(numpy.isnan(bands)))
            self.assert_(not any(numpy.isinf(bands)))
            self.assert_(all(bands >= 0.0))

        # input is a flat spectrum:
        input = [1]*1024
        sr = 44100.
        binfreq = 0.5*sr/(len(input)-1)
        fbands = [8*x*binfreq for x in range(4)]
        expected = [8, 8, 8]
        output = FrequencyBands(frequencyBands=fbands)(input)

        self.assertEqualVector(output, expected)

    def testZero(self):
        # Inputting zeros should return null band energies
        nbands = [0, 100, 200, 300]
        fband = FrequencyBands(frequencyBands=nbands)

        self.assertEqualVector(fband(zeros(1024)), zeros(len(nbands)-1))
        self.assertEqualVector(fband(zeros(10)), zeros(len(nbands)-1))

    def testInvalidParam(self):
        # Test that we must give valid frequency ranges or order
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands':[] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [-1] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [100] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [0, 200, 100] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [0, 200, 200, 400] })

    def testEmpty(self):
        # Test that FrequencyBands on an empty vector should return null band energies
        self.assertComputeFails(FrequencyBands(), [])


suite = allTests(TestFrequencyBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
