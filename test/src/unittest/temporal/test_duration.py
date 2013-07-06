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
import essentia.streaming as es
from random import randint


class TestDuration(TestCase):

    def testEmpty(self):
        input = []
        self.assertEqual(Duration()(input), 0.0)

    def testZero(self):
        input = [0]*100
        self.assertAlmostEqual(Duration()(input), 0.00226757372729)

    def testOne(self):
        input = [0]
        self.assertAlmostEqual(Duration()(input), 2.26757365454e-5)

        input = [100]
        self.assertAlmostEqual(Duration()(input), 2.26757365454e-5)

    def test30Sec(self):
        input = [randint(0, 100) for x in xrange(44100*30)]
        self.assertAlmostEqual(Duration()(input), 30.0)

    def testSampleRates(self):
        self.assertAlmostEqual(Duration(sampleRate=48000)(zeros(100)), 100./48000.)
        self.assertAlmostEqual(Duration(sampleRate=22050)(zeros(100)), 100./22050.)

    def testBadSampleRate(self):
        self.assertConfigureFails(Duration(), { 'sampleRate' : 0 })

    def testFrameStreaming(self):
        gen = VectorInput([0]*100)
        dur = es.Duration()
        pool = Pool()

        gen.data >> dur.signal
        dur.duration >> (pool, 'duration')
        run(gen)

        self.assertAlmostEqual(pool['duration'], 0.00226757372729)

    def testOneStreaming(self):
        gen = VectorInput([ 23 ])
        dur = es.Duration()
        pool = Pool()

        gen.data >> dur.signal
        dur.duration >> (pool, 'duration')
        run(gen)

        self.assertAlmostEqual(pool['duration'], 2.26757365454e-5)

    def test30SecStreaming(self):
        gen = VectorInput([ randint(0, 100) for x in xrange(44100*30) ])
        dur = es.Duration()
        pool = Pool()

        gen.data >> dur.signal
        dur.duration >> (pool, 'duration')
        run(gen)

        self.assertAlmostEqual(pool['duration'], 30.0)

suite = allTests(TestDuration)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
