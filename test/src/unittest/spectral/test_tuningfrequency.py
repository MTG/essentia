#!/usr/bin/env python

from essentia_test import *

class TestTuningFrequency(TestCase):

    def testSinusoid(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_0db.wav'),
                           sampleRate = 44100)()

        fc = FrameCutter(frameSize = 2048,
                         hopSize = 2048)


        windowing = Windowing(type = 'hann')
        spec = Spectrum()

        speaks = SpectralPeaks(sampleRate = 44100,
                               maxPeaks = 1,
                               maxFrequency = 10000,
                               minFrequency = 0,
                               magnitudeThreshold = 0,
                               orderBy = 'magnitude')

        tfreq = TuningFrequency(resolution=1.0)

        while True:
            frame = fc(audio)

            if len(frame) == 0:
                break

            (freqs, mags) = speaks(spec(windowing(frame)))
            (freq, cents) = tfreq(freqs,  mags)

            self.assertAlmostEqual(cents, 0, 4)
            self.assertAlmostEqual(freq, 440.0, 0.5)

    def testEmpty(self):
        (freq,  cents) = TuningFrequency()([],  [])
        self.assertAlmostEqual(freq, 440)
        self.assertAlmostEqual(cents, 0)

    def testZero(self):
        (freq,  cents) = TuningFrequency()([0],  [0])
        self.assertAlmostEqual(freq, 440)
        self.assertAlmostEqual(cents, 0)

    def testSizes(self):
        # Checks whether an exception gets raised,
        # in case the frequency vector has a different size than the magnitude vector
        self.assertComputeFails(TuningFrequency(),  [1],  [1,  2])

    def testInvalidParam(self):
        self.assertConfigureFails(TuningFrequency(), { 'resolution': 0 })

suite = allTests(TestTuningFrequency)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
