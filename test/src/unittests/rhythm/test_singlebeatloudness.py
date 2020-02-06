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



from numpy import *
from essentia_test import *

class TestSingleBeatLoudness(TestCase):

    def testZero(self):
        self.assertEqualVector(SingleBeatLoudness(frequencyBands = [20,150])(zeros(8192)), zeros(2))

    def testEmpty(self):
        self.assertComputeFails(SingleBeatLoudness(), [])

    def testSingle(self):
        self.assertComputeFails(SingleBeatLoudness(), [1])

    def testInvalidInput(self):
        loudness = SingleBeatLoudness(sampleRate=44100,
                                      beatWindowDuration=0.1,
                                      beatDuration=0.05)
        #should fail due to input.size < beatDuration+beatWindowDuration
        self.assertComputeFails(SingleBeatLoudness(), [i for i in range(1000)])

    def testInvalidParam(self):
        self.assertConfigureFails(SingleBeatLoudness(), {'beatWindowDuration':0.05,
                                                         'beatDuration':0.1})

        self.assertConfigureFails(SingleBeatLoudness(), {'beatWindowDuration':0.1,
                                                         'beatDuration':-0.05})

        self.assertConfigureFails(SingleBeatLoudness(), {'beatWindowDuration':-0.1,
                                                         'beatDuration':0.05})

        self.assertConfigureFails(SingleBeatLoudness(), {'sampleRate':-441000})

    def testValidOutput(self):
        # test that it yields valid output (which doesn't mean correct output ;)
        loudness, loudness_bands = SingleBeatLoudness(frequencyBands = [20,150])(array(random.rand(8192)))
        self.assert_(not any(numpy.isnan(loudness)))
        self.assert_(not any(numpy.isinf(loudness)))
        self.assert_(all(array(loudness) >= 0.0))

        self.assert_(not any(numpy.isnan(loudness_bands)))
        self.assert_(not any(numpy.isinf(loudness_bands)))
        self.assert_(all(array(loudness_bands) >= 0.0))

    def testRegression(self):
        # These expected values were obtained from the output
        # of the algorithm SingleBeatLoudness after the commit
        # 3d8fcf5d54844155a5417b1af2399cf87c5ad761
        # For now it is enough to get notified if the algorithm
        # or its dependencies change for any reason, but this
        # does not mean that values are correct as they have
        # not been compared with any other source.
        expected_loudness = 0.2128196358680725
        expected_bands = [0.69004405, 0.2280884, 0.01353478,
                          0.07527339, 0.01737026, 0.05698634]

        filename = join(testdata.audio_dir, 'recorded', 'dubstep.wav')
        fs = 44100.
        audio = MonoLoader(filename=filename, sampleRate=fs)()

        # Take only the 0.2s as it should be a single beat.
        audio = audio[: int(0.2 * fs)]
        loudness, loudness_bands = SingleBeatLoudness()(audio)
        
        self.assertAlmostEqual(loudness, expected_loudness, 1e-6)
        self.assertAlmostEqualVector(loudness_bands, expected_bands, 1e-6)


suite = allTests(TestSingleBeatLoudness)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
