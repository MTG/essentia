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


class TestEnergyBandRatio(TestCase):

    def testRegression(self):
        spectrum = readVector(filename = join(filedir(), 'energybandratio', 'input.txt'))
        expected = readValue(filename = join(filedir(), 'energybandratio', 'output.txt'))

        ebr = EnergyBandRatio(startFrequency = 0.0,
                              stopFrequency = 100.0,
                              sampleRate = 44100)

        self.assertAlmostEqual(ebr(spectrum), expected)


    def testZero(self):
        ebr = EnergyBandRatio(startFrequency = 0.0,
                              stopFrequency = 100.0,
                              sampleRate = 1)
        self.assertEqual(ebr(zeros(1000)), 0)


    def testInvalidParam(self):
        ebr = EnergyBandRatio()
        self.assertConfigureFails(ebr, { 'startFrequency': -25 })
        self.assertConfigureFails(ebr, { 'startFrequency': 100,
                                         'stopFrequency': 80 })

    def testInvalidInput(self):
        self.assertComputeFails(EnergyBandRatio(), [])


suite = allTests(TestEnergyBandRatio)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
