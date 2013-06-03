#!/usr/bin/env python

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
