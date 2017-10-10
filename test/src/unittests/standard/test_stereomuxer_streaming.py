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
from essentia.streaming import AudioLoader, StereoMuxer, StereoDemuxer

class TestStereoMuxer_Streaming(TestCase):

    def testRegression(self):
        filename = join(testdata.audio_dir, 'recorded', 'dubstep.wav')
        loader = AudioLoader(filename=filename)
        demuxer = StereoDemuxer()
        muxer = StereoMuxer()
        p = Pool()

        loader.audio >> demuxer.audio
        loader.audio >> (p, 'original')
        loader.sampleRate >> None
        loader.numberChannels >> None
        loader.md5 >> None
        loader.bit_rate >> None
        loader.codec >> None

        demuxer.left >> muxer.left
        demuxer.right >> muxer.right
        muxer.audio >> (p, 'result')

        run(loader)

        original_l, original_r = zip(*p['original'])
        result_l, result_r = zip(*p['result'])

        self.assertEqualVector(original_l, result_l)
        self.assertEqualVector(original_r, result_r)


    def testEmpty(self):
        filename = join(testdata.audio_dir, 'generated', 'empty', 'empty.aiff')
        loader = AudioLoader(filename=filename)
        demuxer = StereoDemuxer()
        muxer = StereoMuxer()
        p = Pool()

        loader.audio >> demuxer.audio
        loader.sampleRate >> None
        loader.numberChannels >> None
        loader.md5 >> None
        loader.bit_rate >> None
        loader.codec >> None

        demuxer.left >> muxer.left
        demuxer.right >> muxer.right
        muxer.audio >> (p, 'result')

        run(loader)
        self.assertEqual(p.descriptorNames(), [])



suite = allTests(TestStereoMuxer_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
