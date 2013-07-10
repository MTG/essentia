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


class TestFrequencyBands(TestCase):

    def testRegression(self):
        # Simple regression test, comparing normal behaviour
        audio = MonoLoader(filename = join(testdata.audio_dir, 'generated/synthesised/sin440_sweep_0db.wav'),
                           sampleRate = 44100)()

        fft = Spectrum()
        window = Windowing(type = 'hamming')
        fbands = FrequencyBands(sampleRate = 44100)

        for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
            bands = fbands(fft(window(frame)))

            self.assert_(not any(numpy.isnan(bands)))
            self.assert_(not any(numpy.isinf(bands)))
            self.assert_(all(bands >= 0.0))

        # input is a flat spectrum:
        input = [1]*1024
        sr = 44100.
        binfreq = 0.5*sr/(len(input)-1)
        fbands = [8*x*binfreq for x in range(4)]
        expected = [8, 8, 8]
        output = FrequencyBands(frequencyBands=fbands)(input)

        self.assertEqualVector(output, expected)

    def testZero(self):
        # Inputting zeros should return null band energies
        nbands = [0, 100, 200, 300]
        fband = FrequencyBands(frequencyBands=nbands)

        self.assertEqualVector(fband(zeros(1024)), zeros(len(nbands)-1))
        self.assertEqualVector(fband(zeros(10)), zeros(len(nbands)-1))

    def testInvalidParam(self):
        # Test that we must give valid frequency ranges or order
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands':[] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [-1] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [100] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [0, 200, 100] })
        self.assertConfigureFails(FrequencyBands(), { 'frequencyBands': [0, 200, 200, 400] })

    def testEmpty(self):
        # Test that FrequencyBands on an empty vector should return null band energies
        self.assertComputeFails(FrequencyBands(), [])


suite = allTests(TestFrequencyBands)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
