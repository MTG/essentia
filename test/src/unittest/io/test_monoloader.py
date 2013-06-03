#!/usr/bin/env python
#
# Copyright (C) 2006-2008 Music Technology Group (MTG)
#                         Universitat Pompeu Fabra
#
#


from essentia_test import *
from numpy import fabs
audio_dir = join(testdata.audio_dir, 'generated', 'synthesised', 'impulse')
wav_dir = join(audio_dir, 'wav')
ogg_dir = join(audio_dir, 'ogg')
mp3_dir = join(audio_dir, 'mp3')
resamp_dir = join(audio_dir, 'resample')

class TestMonoLoader(TestCase):

    def round(self, val):
        if val >= 0 : return int(val+0.5)
        return int(val-0.5)

    def load(self, filename, downmix, sampleRate):
        return MonoLoader(filename=filename, downmix=downmix, sampleRate=sampleRate)()

    def testInvalidParam(self):
        filename = join(wav_dir, 'impulses_1second_44100_st.wav')
        self.assertConfigureFails(MonoLoader(sampleRate=44100), { 'filename'   : filename,
                                                                  'downmix'    : 'stereo',
                                                                  'sampleRate' : 44100})

        self.assertConfigureFails(MonoLoader(sampleRate=44100), { 'filename'   : filename,
                                                                'downmix'    : 'left',
                                                                'sampleRate' : 0})
        filename = 'unknown.wav'
        self.assertConfigureFails(MonoLoader(), {  'filename' : filename,
                                                    'downmix' : 'left',
                                                 'sampleRate' : 44100})


    def testWav44100(self):
        # files with 9 impulses in each channel
        filename = join(wav_dir, 'impulses_1second_44100_st.wav')
        left = self.load(filename, 'left', 44100);
        right = self.load(filename, 'right', 44100);
        mix = self.load(filename, 'mix', 44100);
        self.assertEqual(self.round(sum(left)), 9)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(self.round(sum(mix)), 9)

    def testWav22050(self):
        # files with 9 impulses in each channel
        filename = join(wav_dir, 'impulses_1second_22050_st.wav')
        left = self.load(filename, 'left', 22050);
        right = self.load(filename, 'right', 22050);
        mix = self.load(filename, 'mix', 22050);
        self.assertEqual(self.round(sum(left)), 9)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(self.round(sum(mix)), 9)

    def testWav48000(self):
        # files with 9 impulses in each channel
        filename = join(wav_dir, 'impulses_1second_48000_st.wav')
        left = self.load(filename, 'left', 48000);
        right = self.load(filename, 'right', 48000);
        mix = self.load(filename, 'mix', 48000);
        self.assertEqual(self.round(sum(left)), 9)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(self.round(sum(mix)), 9)

    def testEmptyWav(self):
        filename = join(testdata.audio_dir, 'generated', 'empty', 'empty.wav')
        self.assertEqualVector(MonoLoader(filename=filename, downmix='left', sampleRate=44100)(), [])

    def testWavLeftRightOffset(self):
        # file with 9 impulses in right channel and 10 in left channel
        dir = join(testdata.audio_dir, 'generated', 'synthesised', 'impulse', 'left_right_offset')
        filename = join(dir, 'impulses_1second_44100.wav')
        left = self.load(filename, 'left', 44100);
        right = self.load(filename, 'right', 44100);
        mix = self.load(filename, 'mix', 44100);
        self.assertEqual(self.round(sum(left)), 10)
        self.assertEqual(self.round(sum(right)), 9)
        self.assertEqual(sum(mix), 9.5) # 0.5*left + 0.5*right

###############
# #mp3
###############

    def sum(self, l):
        result = 0.0
        noisefloor = 0.003
        for i in range(len(l)):
            if fabs(l[i]) > noisefloor:
               result+= l[i]
        return self.round(result)

    def testMp344100(self):
        # files with 9 impulses in each channel
        filename = join(mp3_dir, 'impulses_1second_44100_st.mp3')
        left = self.load(filename, 'left', 44100);
        right = self.load(filename, 'right', 44100);
        mix = self.load(filename, 'mix', 44100);
        self.assertEqual(self.sum(left), 9)
        self.assertEqual(self.sum(right), 9)
        self.assertEqual(self.sum(mix), 9)

    def testMp322050(self):
        # files with 9 impulses in each channel
        filename = join(mp3_dir, 'impulses_1second_22050_st.mp3')
        left = self.load(filename, 'left', 22050);
        right = self.load(filename, 'right', 22050);
        mix = self.load(filename, 'mix', 22050);
        self.assertEqual(self.sum(left), 9)
        self.assertEqual(self.sum(right), 9)
        self.assertEqual(self.sum(mix), 9)

    def testMp348000(self):
        # files with 9 impulses in each channel
        filename = join(mp3_dir, 'impulses_1second_48000_st.mp3')
        left = self.load(filename, 'left', 48000);
        right = self.load(filename, 'right', 48000);
        mix = self.load(filename, 'mix', 48000);
        self.assertEqual(self.sum(left), 9)
        self.assertEqual(self.sum(right), 9)
        self.assertEqual(self.sum(mix), 9)

###############
# #OGG
###############

    def testOgg44100(self):
        filename = join(ogg_dir, 'impulses_1second_44100_st.ogg')
        left = self.load(filename, 'left', 44100);
        right = self.load(filename, 'right', 44100);
        mix = self.load(filename, 'mix', 44100);
        self.assertEqual(abs(self.sum(left)),  9)
        self.assertEqual(abs(self.sum(right)), 9)
        self.assertEqual(abs(self.sum(mix)),   9)

        if self.sum(left) < 0:
            print 'WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...'

    def testOgg22050(self):
        # files with 9 impulses in each channel
        filename = join(ogg_dir, 'impulses_1second_22050_st.ogg')
        left = self.load(filename, 'left', 22050);
        right = self.load(filename, 'right', 22050);
        mix = self.load(filename, 'mix', 22050);
        self.assertEqual(abs(self.sum(left)),  9)
        self.assertEqual(abs(self.sum(right)), 9)
        self.assertEqual(abs(self.sum(mix)),   9)

        if self.sum(left) < 0:
            print 'WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...'

    def testOgg48000(self):
        # files with 9 impulses in each channel
        filename = join(ogg_dir, 'impulses_1second_48000_st.ogg')
        left = self.load(filename, 'left', 48000);
        right = self.load(filename, 'right', 48000);
        mix = self.load(filename, 'mix', 48000);
        self.assertEqual(abs(self.sum(left)),  9)
        self.assertEqual(abs(self.sum(right)), 9)
        self.assertEqual(abs(self.sum(mix)),   9)

        if self.sum(left) < 0:
            print 'WARNING: Essentia uses a version of FFMpeg that does reverse decoding of Ogg files...'

    def testDownSampling(self):
        # files of 30s with impulses at every sample
        # from 44100 to 22050
        filename = join(resamp_dir, 'impulses_1samp_44100.wav')
        left = self.load(filename, 'left', 22050);
        self.assertAlmostEqual(sum(left), 30.*22050, 1e-4)
        # from 48000 to 44100
        filename = join(resamp_dir, 'impulses_1samp_48000.wav')
        left = self.load(filename, 'left', 44100);
        self.assertAlmostEqual(sum(left), 30.*44100, 1e-4)
        # from 48000 to 22050
        left = self.load(filename, 'left', 22050);
        self.assertAlmostEqual(sum(left), 30.*22050, 1e-4)

    def testUpSampling(self):
        # from 44100 to 48000
        filename = join(resamp_dir, 'impulses_1samp_44100.wav')
        left = self.load(filename, 'right', 48000);
        self.assertAlmostEqual(sum(left), 30.*48000, 1e-4)
        # from 22050 to 44100
        filename = join(resamp_dir, 'impulses_1samp_22050.wav')
        left = self.load(filename, 'right', 44100);
        self.assertAlmostEqual(sum(left), 30.*44100, 1e-4)
        # from 22050 to 48000
        left = self.load(filename, 'right', 48000);
        self.assertAlmostEqual(sum(left), 30.*48000, 1e-4)

    def testInvalidFilename(self):
        self.assertConfigureFails(MonoLoader(),{'filename':'unknown.wav'})

    def testResetStandard(self):
        audiofile = join(testdata.audio_dir,'recorded','britney.wav')
        loader = MonoLoader(filename=audiofile)
        audio1 = loader();
        audio2 = loader();
        loader.reset();
        audio3 = loader();
        self.assertAlmostEqualVector(audio3, audio1)
        self.assertEqualVector(audio2, audio1)

    def testLoadMultiple(self):
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = MonoLoader(filename=filename)
        audio1 = algo()
        audio2 = algo()
        audio3 = algo()
        self.assertEquals(len(audio1), 441000);
        self.assertEquals(len(audio2), 441000);
        self.assertEquals(len(audio3), 441000);
        self.assertEqualVector(audio2, audio1)
        self.assertEqualVector(audio2, audio3)





suite = allTests(TestMonoLoader)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
