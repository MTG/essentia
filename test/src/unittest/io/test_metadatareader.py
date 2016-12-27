#!/usr/bin/env python

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

    def testFlac(self):
        result = MetadataReader(filename = join(self.audioDir, 'test.flac'))()
        tagsPool = result[7]
        tags = tagsPool.descriptorNames() + [tagsPool[t][0] for t in tagsPool.descriptorNames()]
        
        self.assertEqualVector(result[:7], ('test flac', 'mtg', 'essentia', '', 'Thrash Metal', '01', '2009'))
        
        # FIXME: Taglib 1.11.0 on OSX outputs bitrate inconsistent with 1.9.1 on Linux for FLAC and OGG
        # It might be due to different versions of Taglib or due to different platforms (we have not tested)
        # Therefore accept both bitrates as correct

        self.assertEqualVector([result[8]] + list(result[10:]), (5, 44100, 4))
        self.assertTrue(result[9] == 2201 or result[9] == 2202)

        self.assertEqualVector(
                tags, 
                ['metadata.tags.album', 'metadata.tags.artist', 'metadata.tags.composer', 'metadata.tags.copyright', 
                 'metadata.tags.date', 'metadata.tags.description', 'metadata.tags.discnumber', 'metadata.tags.genre', 
                 'metadata.tags.performer', 'metadata.tags.title', 'metadata.tags.tracknumber', 'metadata.tags.tracktotal', 
                 'essentia', 'mtg', 'roberto.toscano', 'mtg.upf.edu', '2009', 'This is not thrash metal', '01', 'Thrash Metal', 
                 'roberto.toscano', 'test flac', '01', '01']
            )

    def testOgg(self):
        result = MetadataReader(filename = join(self.audioDir, 'test.ogg'))()
        tagsPool = result[7]
        tags = tagsPool.descriptorNames() + [tagsPool[t][0] for t in tagsPool.descriptorNames()]

        self.assertEqualVector(result[:7], ('test ogg', 'mtg', 'essentia', 'this is not psychadelic', 'Psychadelic', '01', '2009'))

        # see the FIXME note above
        self.assertEqualVector([result[8]] + list(result[10:]), (5, 44100, 1))
        self.assertTrue(result[9] == 96 or result[9] == 20)

        self.assertEqualVector(
                tags, 
                ['metadata.tags.album', 'metadata.tags.artist', 'metadata.tags.comment', 'metadata.tags.composer', 
                 'metadata.tags.copyright', 'metadata.tags.date', 'metadata.tags.description', 'metadata.tags.discnumber', 
                 'metadata.tags.genre', 'metadata.tags.performer', 'metadata.tags.title', 'metadata.tags.tracknumber', 
                 'metadata.tags.tracktotal', 'essentia', 'mtg', 'this is not psychadelic', 'roberto.toscano', 'mtg.upf.edu', 
                 '2009', 'this is not psychadelic', '1', 'Psychadelic', 'roberto.toscano', 'test ogg', '01', '01']
            )

    def testMp3(self):
        result = MetadataReader(filename = join(self.audioDir, 'test.mp3'))()
        tagsPool = result[7]
        tags = tagsPool.descriptorNames() + [tagsPool[t][0] for t in tagsPool.descriptorNames()]

        self.assertEqualVector(result[:7], ('test sound', 'mtg', 'essentia', 'this is not reggae', 'Reggae', '01', '2009'))
        self.assertEqualVector(result[8:], (5, 128, 44100, 1))
        self.assertEqualVector(
                tags, 
                ['metadata.tags.album', 'metadata.tags.artist', 'metadata.tags.comment', 'metadata.tags.date', 
                 'metadata.tags.genre', 'metadata.tags.title', 'metadata.tags.tracknumber', 'essentia', 'mtg', 
                 'this is not reggae', '2009', 'Reggae', 'test sound', '01']
            )

    def testApe(self):
        result = MetadataReader(filename = join(self.audioDir, 'test.ape'))()
        tagsPool = result[7]
        tags = tagsPool.descriptorNames() + [tagsPool[t][0] for t in tagsPool.descriptorNames()]

        self.assertEqualVector(result[:7], ('ape test file', 'mtg', 'essentia', 'this is not porn', 'Porn Groove', "01/01", "2009"))
        self.assertEqualVector(result[8:], (5, 722, 44100, 1))
        self.assertEqualVector(
                tags, 
                ['metadata.tags.album', 'metadata.tags.artist', 'metadata.tags.comment', 'metadata.tags.composer', 
                 'metadata.tags.copyright', 'metadata.tags.date', 'metadata.tags.genre', 'metadata.tags.original artist', 
                 'metadata.tags.part', 'metadata.tags.title', 'metadata.tags.tracknumber', 'essentia', 'mtg', 'this is not porn', 
                 'roberto.toscano', 'mtg.upf.edu', '2009', 'Porn Groove', 'roberto.toscano', '1', 'ape test file', '01/01']
            )

    def testPCM(self):
        result = MetadataReader(filename = join(testdata.audio_dir, 'recorded', 'musicbox.wav'), failOnError=True)()
        
        self.assertTrue(not len(result[7].descriptorNames()))
        self.assertEqualVector(result[:7], ('', '', '', '', '', '', ''))
        self.assertEqualVector(result[8:], (45, 1444, 44100, 2))

    def testFailOnError(self):
        self.assertComputeFails(
            MetadataReader(filename = join(self.audioDir, 'random_file_that_doesnt_exist.ape'), failOnError=True))
        
        result = MetadataReader(filename = join(self.audioDir, 'random_file_that_doesnt_exist.ape'), failOnError=False)()

        self.assertTrue(result[7].descriptorNames() == [])
        self.assertEqualVector(result[:7], ('', '', '', '', '', '', ''))
        self.assertEqualVector(result[8:], (0, 0, 0, 0))

    def testUnicode(self):
        result = MetadataReader(filename = join(self.audioDir, 'test-unicode.flac'))()
        self.assertEqualVector(result[:7], ('test flac &n"jef\';:/?.>,<-_=+)(*&^%$#@!~`', '?mtg $#@!$"&', '', '', '', '', ''))

    def testEmpty(self):
        self.assertComputeFails(MetadataReader())

    def testEmptyTags(self):
        result = MetadataReader(filename = join(self.audioDir, 'empty.mp3'))()

        self.assertEqualVector(result[:7], ('', '', '', '', '', '', ''))
        self.assertTrue(result[7].descriptorNames() == [])
        self.assertEqual(result[8], 0)
        # Outputs [-3:] correspond to bitrate, samplerate and channels, and will differ depending on taglib version
        # Therefore, not testing them.

suite = allTests(TestMetadataReader)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
