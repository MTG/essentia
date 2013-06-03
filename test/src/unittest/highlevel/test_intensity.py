#!/usr/bin/env python

from essentia_test import *


class TestIntensity(TestCase):

    def testEmpty(self):
        self.assertComputeFails(Intensity(), [])

    def testSilence(self):
        audio = [0]*(44100*10) # 10 sec silence
        self.assertEqual(Intensity()(audio), -1) # silence is relaxing isn't it

    def testDiffSampleRates(self):
        algo44 = Intensity(sampleRate=44100)
        algo22 = Intensity(sampleRate=22050)

        filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        britney44 = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        filename = join(testdata.audio_dir, 'recorded', 'britney22050.wav')
        britney22 = MonoLoader(filename=filename, downmix='left', sampleRate=22050)()

        self.assertEqual(algo44(britney44), algo22(britney22))

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        britneyIntensity = Intensity()(audio)

        filename = join(testdata.audio_dir, 'recorded', '01-Allegro__Gloria_in_excelsis_Deo_in_D_Major.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        gloriaIntensity = Intensity()(audio)

        filename = join(testdata.audio_dir, 'recorded', 'roxette.wav')
        audio = MonoLoader(filename=filename, downmix='left', sampleRate=44100)()
        roxetteIntensity = Intensity()(audio)

        self.assertTrue(britneyIntensity > gloriaIntensity)
        self.assertTrue(britneyIntensity >= roxetteIntensity)
        self.assertTrue(roxetteIntensity > gloriaIntensity)


suite = allTests(TestIntensity)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
