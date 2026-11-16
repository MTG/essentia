#!/usr/bin/env python

import os
from essentia_test import *
from essentia.standard import AudioCommonsTimbralModels, MonoLoader
import numpy as np

try:
    from .example_sounds_data import example_sounds_data
except ImportError:
    from example_sounds_data import example_sounds_data


class TestAudioCommonsTimbralModels(TestCase):

    def setUp(self):
        self.extractor = AudioCommonsTimbralModels()

    def _get_audio_path(self, rel_path):
        if os.path.exists(join(testdata.audio_dir, rel_path)):
            return join(testdata.audio_dir, rel_path)
        if os.path.exists(join('test', 'audio', rel_path)):
            return join('test', 'audio', rel_path)
        repo_root = os.path.abspath(join(os.path.dirname(__file__), '..', '..', '..'))
        return join(repo_root, 'test', 'audio', rel_path)

    def testFiles(self):
        for sound_data in example_sounds_data:
            audio_path = self._get_audio_path(sound_data["path"])
            audio = MonoLoader(filename=audio_path, sampleRate=44100)()
            (brightness, depth, hardness, roughness,
             sharpness, warmth, boominess) = self.extractor(audio)

            self.assertAlmostEqual(brightness, sound_data["brightness"], precision=1)
            self.assertAlmostEqual(depth, sound_data["depth"], precision=1)
            self.assertAlmostEqual(hardness, sound_data["hardness"], precision=1)
            self.assertAlmostEqual(roughness, sound_data["roughness"], precision=1)
            self.assertAlmostEqual(sharpness, sound_data["sharpness"], precision=1)
            self.assertAlmostEqual(warmth, sound_data["warmth"], precision=1)
            self.assertAlmostEqual(boominess, sound_data["boominess"], precision=1)

    def testSelectiveGating(self):
        sound_data = example_sounds_data[0]
        audio_path = self._get_audio_path(sound_data["path"])
        audio = MonoLoader(filename=audio_path, sampleRate=44100)()

        extractor = AudioCommonsTimbralModels(computeHardness=False)
        (brightness, depth, hardness, roughness,
         sharpness, warmth, boominess) = extractor(audio)

        self.assertEqual(hardness, 0.0)
        self.assertAlmostEqual(brightness, sound_data["brightness"], precision=1)
        self.assertAlmostEqual(depth, sound_data["depth"], precision=1)
        self.assertAlmostEqual(roughness, sound_data["roughness"], precision=1)
        self.assertAlmostEqual(sharpness, sound_data["sharpness"], precision=1)
        self.assertAlmostEqual(warmth, sound_data["warmth"], precision=1)
        self.assertAlmostEqual(boominess, sound_data["boominess"], precision=1)

    def testSilenceInput(self):
        silence = np.zeros(44100, dtype=np.float32)
        (brightness, depth, hardness, roughness,
         sharpness, warmth, boominess) = self.extractor(silence)

        self.assertEqual(brightness, 0.0)
        self.assertEqual(depth, 0.0)
        self.assertEqual(hardness, 0.0)
        self.assertEqual(roughness, 0.0)
        self.assertEqual(sharpness, 0.0)
        self.assertEqual(warmth, 0.0)
        self.assertEqual(boominess, 0.0)


suite = allTests(TestAudioCommonsTimbralModels)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
