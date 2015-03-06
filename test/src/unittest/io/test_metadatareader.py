#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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
import os


def taglibVersion():
    if sys.platform == 'linux2':
        return os.popen('dpkg -l libtag*dev | grep ii').read().split('\n')[0].split()[2]
    if sys.platform == 'darwin':
        return os.popen('taglib-config --version').read().split('\n')[0]

    return '??'

class TestMetadataReader(TestCase):

    audioDir = join(testdata.audio_dir, 'generated', 'metadata')

    def testOgg(self):
        result = MetadataReader(filename = join(self.audioDir, 'test.ogg'))()
        result = result[:7] + result[8:] # FIXME: ignoring Pool output
        self.assertEqualVector(result, 
                               ('test ogg', 'mtg', 'essentia', 'this is not psychadelic', 'Psychadelic', '01', '2009', 5, 96, 44100, 1))

    def testMp3(self):
        result = MetadataReader(filename = join(self.audioDir, 'test.mp3'))()
        result = result[:7] + result[8:] # FIXME: ignoring Pool output

        self.assertEqualVector(result,
                               ('test sound', 'mtg', 'essentia', 'this is not reggae', 'Reggae', '01', '2009', 5, 128, 44100, 1))

    def testFlac(self):
        print MetadataReader(filename = join(self.audioDir, 'test.flac'))()[:7]

        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test.flac'))()[:7],
                ('test flac', 'mtg', 'essentia', 'This is not thrash metal', 'Thrash Metal', '01', '2009'))
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test.flac'))()[8:],
                (5, 2201, 44100, 4))

    def testPCM(self):
        result = MetadataReader(filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav'), failOnError=True)()
        result = result[:7] + result[8:] # FIXME: ignoring Pool output
        self.assertEqualVector(result, ('', '', '', '', '', '', '', 45, 1444, 44100, 2))

    def testFailOnError(self):
        if taglibVersion() < '1.7':
            self.assertComputeFails(
                MetadataReader(filename = join(self.audioDir, 'test.ape'), failOnError=True))
        else:
            # FIXME We also need to check if pool output is correct...
            self.assertEqualVector(MetadataReader(filename = join(self.audioDir, 'test.ape'), failOnError=True)()[:7],
                                   ('ape test file', 'mtg', 'essentia', 'this is not porn', 'Porn Groove', "01/01", "2009"))
            self.assertEqualVector(MetadataReader(filename = join(self.audioDir, 'test.ape'), failOnError=True)()[8:],
                                   (5, 722, 44100, 1))

    def testUnicode(self):
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test-unicode.flac'))(),
                ('test flac &n"jef\';:/?.>,<-_=+)(*&^%$#@!~`', '?mtg $#@!$"&', '', '', '', 0, 0, 5, 2201, 44100, 4))

    def testEmpty(self):
        self.assertComputeFails(MetadataReader())

    def testEmptyTags(self):
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'empty.mp3'))()[:7],
                ('', '', '', '', '', '', ''))
        self.assertEqual(MetadataReader(filename = join(self.audioDir, 'empty.mp3'))()[8], 0)
        self.assertTrue(MetadataReader(filename = join(self.audioDir, 'empty.mp3'))()[7].descriptorNames() == [])
        # Outputs [-3:] correspond to bitrate, samplerate and channels, and will differ depending on taglib version
        # Therefore, not testing them.


suite = allTests(TestMetadataReader)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
