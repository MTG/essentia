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
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test.ogg'))(),
                ('test ogg', 'mtg', 'essentia', 'this is not psychadelic', 'Psychadelic', 1, 2009, 5, 96, 44100, 1))

    def testMp3(self):
        # TagLib 1.4 doesn't correctly read the number of channels for mp3, do not run the test on that case
        try:
            if taglibVersion() < '1.5':
                self.assertEqualVector(
                    MetadataReader(filename = join(self.audioDir,'test.mp3'))()[:-1],
                    ('test sound', 'mtg', 'essentia', 'this is not reggae', 'Reggae', 1, 2009, 5, 128, 44100))
                return
        except:
            pass

        # TagLib 1.5 in os x doesn't correctly read the number of channels for mp3, do not run the test on that case
        if taglibVersion() <= '1.5':
            self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir,'test.mp3'))()[:-1],
                ('test sound', 'mtg', 'essentia', 'this is not reggae', 'Reggae', 1, 2009, 5, 128, 44100))
        else:
            self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test.mp3'))(),
                ('test sound', 'mtg', 'essentia', 'this is not reggae', 'Reggae', 1, 2009, 5, 128, 44100, 1))

    def testFlac(self):
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test.flac'))(),
                ('test flac', 'mtg', 'essentia', 'This is not thrash metal', 'Thrash Metal', 1, 2009, 5, 2201, 44100, 4))

    def testPCM(self):
        if taglibVersion() < '1.7':
            self.assertEqualVector(
                MetadataReader(filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav'), failOnError=True)(),
                               ('', '', '', '', '', 0, 0, 0, 1411, 44100, 2))
        else:
            self.assertEqualVector(
                MetadataReader(filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav'), failOnError=True)(),
                               ('', '', '', '', '', 0, 0, 45, 1444, 44100, 2))

    def testFailOnError(self):
        if taglibVersion() < '1.7':
            self.assertComputeFails(
                MetadataReader(filename = join(self.audioDir, 'test.ape'), failOnError=True))
        else:
            self.assertEqualVector(MetadataReader(filename = join(self.audioDir, 'test.ape'), failOnError=True)(),
                                   ('ape test file', 'mtg', 'essentia', 'this is not porn', 'Porn Groove', 1, 2009, 5, 722, 44100, 1))

    def testUnicode(self):
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'test-unicode.flac'))(),
                ('test flac &n"jef\';:/?.>,<-_=+)(*&^%$#@!~`', '?mtg $#@!$"&', '', '', '', 0, 0, 5, 2201, 44100, 4))

    def testEmpty(self):
        self.assertComputeFails(MetadataReader())

    def testEmptyTags(self):
        # results for this type of file will differ depending on taglib's
        # version:
        # for version < 1.5: ('""', '""', '""', '""', '""', 0, 0, 0, 0, 0, 0))
        # for version >= 1.5: ('""', '""', '""', '""', '""', 0, 0, 0, 128, 44100, 1))
        # therefore, only testing for the first fields:
        self.assertEqualVector(
                MetadataReader(filename = join(self.audioDir, 'empty.mp3'))()[:-3],
                ('', '', '', '', '', 0, 0, 0))


suite = allTests(TestMetadataReader)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
