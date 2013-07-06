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

class TestBpmRubato(TestCase):

    def PulseTrain(self, period, phase, length):
        beats = []
        tick = phase

        while (tick < length):
            beats.append(tick)
            tick += 60./period
        return beats

    def IncreasingPulseTrain(self, period, phase, length):
        beats = []
        tick = phase
        while (tick < length):
            beats.append(tick)
            period += 1
            tick += 60./period
        return beats

    def DecreasingPulseTrain(self, period, phase, length):
        beats = []
        tick = phase
        while (tick < length):
            beats.append(tick)
            period -= 1
            tick += 60./period
        return beats

    def testRegression(self):
        # build a 60s pulse train at 100bpm starting at 0.3s:
        # (not a rubato region)
        beats = self.PulseTrain(100, 0.3, 60.)
        start, stop = BpmRubato()(beats)
        self.assertEqualVector(start, [])
        self.assertEqualVector(stop, [])
        self.assertTrue( beats[-1] < 60 )

        # add a 60s pulse train at 140bpm starting 0.6s later:
        # (this is a rubato region)
        phase = beats[-1] + 60/100.
        expectedStart = [beats[-3]]
        newBeats = self.PulseTrain(140, phase, phase+60.)
        expectedStop = [newBeats[2]]
        beats += newBeats
        start, stop = BpmRubato()(beats)
        self.assertAlmostEqualVector(start, expectedStart, 1e-1)
        self.assertAlmostEqualVector(stop, expectedStop, 1e-1)
        self.assertTrue( beats[-1] < 120 )

        # add 60s more of a pulse train at 140bpm starting 0.3s later
        # (not a rubato region)
        phase = 0.3 + beats[-1]
        beats += self.PulseTrain(140, phase, phase+60.)
        start, stop = BpmRubato()(beats)
        self.assertAlmostEqualVector(start, expectedStart, 1e-1)
        self.assertAlmostEqualVector(stop, expectedStop, 1e-1)
        self.assertEqual(True, beats[-1] < 180.6 )

        # add a 10s pulse train at 100bpm starting 0.3s later:
        # slowing down (this is a rubato region)
        phase = 0.3 + beats[-1]
        newBeats = self.PulseTrain(100, phase, phase+10.)
        expectedStart += [beats[-3]]
        expectedStop += [newBeats[2]]
        beats += newBeats
        start, stop = BpmRubato()(beats)
        self.assertEqual(len(start), len(stop))
        self.assertAlmostEqualVector(start, expectedStart, 1e-1)
        self.assertAlmostEqualVector(stop, expectedStop, 1e-1)
        self.assertTrue( beats[-1] < 190.6 )

        # add 10s more of increasing train at 110bpm
        # (this is not a rubato region because of the gradual change)
        phase = 0.3 + beats[-1]
        newBeats = self.IncreasingPulseTrain(110, phase, phase+10.)
        beats += newBeats

        start, stop = BpmRubato()(beats)

        self.assertEqual(len(start), len(stop))
        self.assertAlmostEqualVector(start, expectedStart, 1e-1)
        self.assertAlmostEqualVector(stop, expectedStop, 1e-1)
        self.assertTrue( beats[-1] < 200.6 )

        # add 10s more of decreasing train at 110bpm
        # (this is a rubato region, its gradual like the one before, but I'm
        # not sure why its registered as a rubato region)
        phase = 0.3 + beats[-1]
        newBeats = self.DecreasingPulseTrain(110, phase, phase+10.)
        expectedStart += [beats[-3]]
        expectedStop += [newBeats[2]]

        beats += newBeats

        start, stop = BpmRubato()(beats)

        self.assertEqual(len(start), len(stop))
        self.assertAlmostEqualVector(start, expectedStart, 1e-1)
        self.assertAlmostEqualVector(stop, expectedStop, 1e-1)
        self.assertTrue( beats[-1] < 210.6 )

        # add more pulses, but no phase offset:
        # this fails due to beats not being in ascending time
        phase = 0.3
        beats += self.PulseTrain(140, phase, 60.)
        self.assertComputeFails(BpmRubato(), beats)

    def testConstantInput(self):
        beats = ones(100)
        self.assertComputeFails(BpmRubato(),(beats))
        beats = zeros(100)
        self.assertComputeFails(BpmRubato(),(beats))

    def testInvalidParam(self):
        # Test that we must give valid frequency ranges:
        self.assertConfigureFails(BpmRubato(),{'tolerance':2})
        self.assertConfigureFails(BpmRubato(),{'tolerance':-2})
        self.assertConfigureFails(BpmRubato(), { 'longRegionsPruningTime': -1 })
        self.assertConfigureFails(BpmRubato(), { 'shortRegionsMergingTime': -1 })

    def testDescendingBeats(self):
        beats = range(10)
        beats.reverse()
        self.assertComputeFails(BpmRubato(), beats)

    def testEmpty(self):
        beats = []
        start, stop = BpmRubato()(beats)
        self.assertEqualVector(start, [])
        self.assertEqualVector(stop, [])


suite = allTests(TestBpmRubato)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
