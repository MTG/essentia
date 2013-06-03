#!/usr/bin/env python

from essentia_test import *


class TestDanceability(TestCase):
    def testEmpty(self):
        self.assertEqual(Danceability()([]), 0)

    def testOne(self):
        self.assertEqual(Danceability()([10]), 0)

    def testSilence(self):
        self.assertEqual(Danceability()([0]*44100), 0)

    def testRegression(self):
        audio = MonoLoader(filename = join(testdata.audio_dir, 'recorded', 'roxette.wav'),
                            downmix='left', sampleRate=44100)()
        self.assertAlmostEqual(Danceability()(audio), 1.25313031673, 5e-3)

    def testSanity(self):
        filename = join(testdata.audio_dir, 'recorded', \
                        '01-Allegro__Gloria_in_excelsis_Deo_in_D_Major.wav')

        gloriaAudio = MonoLoader(filename=filename,
                                 downmix='left',
                                 sampleRate=44100)()
        filename = join(testdata.audio_dir, 'recorded', 'britney.wav')
        britneyAudio = MonoLoader(filename=filename,
                                  downmix='left',
                                  sampleRate=44100)()

        gloriaDanceability = Danceability()(gloriaAudio)
        britneyDanceability = Danceability()(britneyAudio)

        self.assertTrue(gloriaDanceability < britneyDanceability)

    def testMinBiggerThanMaxTau(self):
        self.assertConfigureFails(Danceability(), {'minTau':1000, 'maxTau':500})

    def testZero(self):
        input = [0]*100000

        self.assertAlmostEqual(Danceability()(input), 0, 1e-2)


suite = allTests(TestDanceability)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
