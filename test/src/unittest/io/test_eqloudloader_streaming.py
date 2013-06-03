#!/usr/bin/env python

from essentia_test import *
from essentia.streaming import EasyLoader, EqloudLoader
import sys
class TestEqloudLoader_Streaming(TestCase):

    def load(self, inputSampleRate, outputSampleRate,
                   eqloudfilename, normalfilename,
                   downmix, replayGain, startTime, endTime):

        eqloudloader = EqloudLoader(filename=normalfilename,
                                    sampleRate = outputSampleRate,
                                    downmix = downmix,
                                    startTime = startTime,
                                    endTime = endTime,
                                    replayGain = replayGain)

        easyloader = EasyLoader(filename=eqloudfilename,
                                sampleRate = outputSampleRate,
                                downmix = downmix,
                                startTime = startTime,
                                endTime = endTime,
                                replayGain = replayGain)
        pool = Pool()

        easyloader.audio >> (pool, 'expected')
        run(easyloader)

        eqloudloader.audio >> (pool, 'eqloud')
        run(eqloudloader)

        for val1, val2 in zip(pool['eqloud'][outputSampleRate:],
                              pool['expected'][outputSampleRate:]):
              self.assertAlmostEqual(val1-val2, 0, 5e-3)


    def testNoResample(self):
        eqloud=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds_eqloud.wav')
        normal=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds.wav')
        self.load(44100, 44100, eqloud, normal, "left" , -6.0, 0., 30.);
        self.load(44100, 44100, eqloud, normal, "left", -6.0, 3.35, 5.68);
        self.load(44100, 44100, eqloud, normal, "left"  , -6.0, 0.169, 8.333);

    def testResample(self):
        eqloud=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds_eqloud.wav')
        normal=join(testdata.audio_dir,'generated','doublesize','sin_30_seconds.wav')
        self.load(44100, 48000, eqloud, normal, "left", -6.0, 3.35, 5.68);
        self.load(44100, 32000, eqloud, normal, "left", -6.0, 3.35, 5.68);



    def testInvalidParam(self):
        filename = join(testdata.audio_dir, 'generated','synthesised','impulse','resample',
                        'impulses_1samp_44100.wav')
        self.assertConfigureFails(EqloudLoader(), {'filename':'unknown.wav'})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'downmix' : 'stereo'})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'sampleRate' : 0})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'startTime' : -1})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'endTime' : -1})
        self.assertConfigureFails(EqloudLoader(), {'filename':filename, 'startTime':10, 'endTime' : 1})

    def testResetStandard(self):
        from essentia.standard import EqloudLoader as stdEqloudLoader
        audiofile = join(testdata.audio_dir,'recorded','britney.wav')
        loader = stdEqloudLoader(filename=audiofile, endTime=31)
        audio1 = loader();
        audio2 = loader();
        loader.reset();
        audio3 = loader();
        self.assertAlmostEqualVector(audio3, audio1)
        self.assertEqualVector(audio2, audio1)

    def testLoadMultiple(self):
        from essentia.standard import EqloudLoader as stdEqloudLoader
        aiffpath = join('generated','synthesised','impulse','aiff')
        filename = join(testdata.audio_dir,aiffpath,'impulses_1second_44100.aiff')
        algo = stdEqloudLoader(filename=filename)
        audio1 = algo()
        audio2 = algo()
        audio3 = algo()
        self.assertEquals(len(audio1), 441000);
        self.assertEquals(len(audio2), 441000);
        self.assertEquals(len(audio3), 441000);
        self.assertEqualVector(audio2, audio1)
        self.assertEqualVector(audio2, audio3)



suite = allTests(TestEqloudLoader_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
