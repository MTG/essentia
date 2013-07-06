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
from essentia.streaming import LogAttackTime as sLogAttackTime

class TestLogAttackTime_Streaming(TestCase):

    def testEmpty(self):
        gen = VectorInput([])
        logAttack = sLogAttackTime()
        accu = RealAccumulator()
        p = Pool()

        gen.data >> accu.data
        accu.array >> logAttack.signal
        logAttack.logAttackTime >> (p, 'logAttackTime')

        run(gen)

        self.assertEqual(p.descriptorNames(), [])


    def testRegression(self):
        # triangle input
        input = [float(i) for i in range(22050)]
        input.reverse()
        input += [float(i) for i in range(22050)]

        gen = VectorInput(input)
        logAttack = sLogAttackTime()
        accu = RealAccumulator()
        p = Pool()

        gen.data >> accu.data
        accu.array >> logAttack.signal
        logAttack.logAttackTime >> (p, 'logAttackTime')

        run(gen)

        self.assertAlmostEqual(p['logAttackTime'][0], LogAttackTime()(input))


suite = allTests(TestLogAttackTime_Streaming)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
