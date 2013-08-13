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


class TestPoolAggregator(TestCase):

    def testAggregateReal(self):
        p = Pool({ 'foo': [ 1, 1, 2, 3, 5, 8, 13, 21, 34 ] })

        pAgg = PoolAggregator(defaultStats=['mean', 'min', 'max', 'median', 'var', 'dmean', 'dvar', 'dmean2', 'dvar2'])

        results = pAgg(p)

        self.assertAlmostEqual(results['foo.mean'], 9.77777777778)
        self.assertAlmostEqual(results['foo.median'], 5)
        self.assertAlmostEqual(results['foo.min'], 1)
        self.assertAlmostEqual(results['foo.max'], 34)
        self.assertAlmostEqual(results['foo.var'], 112.172839506)
        self.assertAlmostEqual(results['foo.dmean'], 4.125)
        self.assertAlmostEqual(results['foo.dvar'], 17.109375)
        self.assertAlmostEqual(results['foo.dmean2'], 1.85714285714)
        self.assertAlmostEqual(results['foo.dvar2'], 2.40816326531, 1e-6)


    def testAggregateExceptions(self):
        # default aggregates
        defaultStats = ['mean', 'var', 'min', 'max']

        # and the exceptions
        exceptions = {'foo.bar': ['min']}

        # prepare a pool
        p = Pool({ 'foo.bar': [ 1.1, 2.2, 3.3 ],
                   'bar.foo': [ -6.9, 6.9 ] })

        # configure and aggregate
        results = PoolAggregator(defaultStats=defaultStats, exceptions=exceptions)(p)

        # verify that mean, var, min, max were computed for (bar, foo), but only
        # min was computed for (foo, bar)
        expectedKeys = ['foo.bar.min', 'bar.foo.min', 'bar.foo.max',
                        'bar.foo.mean', 'bar.foo.var']
        resultingKeys = results.descriptorNames()

        self.assertEqualVector(sorted(resultingKeys), sorted(expectedKeys))

        self.assertAlmostEqual(results['foo.bar.min'], 1.1)

        self.assertAlmostEqual(results['bar.foo.min'], -6.9)
        self.assertAlmostEqual(results['bar.foo.max'], 6.9)
        self.assertAlmostEqual(results['bar.foo.mean'], 0.0)
        self.assertAlmostEqual(results['bar.foo.var'], 47.61)


    def testStringAggregation(self):
        p = Pool({ 'foo': [ 'foo', 'bar' ] })

        results = PoolAggregator()(p)

        self.assertEqualVector(results['foo'], ['foo', 'bar'])
        self.assertEqualVector(results.descriptorNames(), ['foo'])


    def testMatrixRealAggregation(self):
        p = Pool()
        p.add('foo', [1.1, 2.2, 3.3])
        p.add('foo', [4.4, 5.5, 6.6])
        p.add('foo', [7.7, 8.8, 9.9])

        defaultStats = ['mean', 'min', 'max', 'median', 'var', 'dmean', 'dvar', 'dmean2', 'dvar2']
        results = PoolAggregator(defaultStats=defaultStats)(p)

        self.assertAlmostEqualVector(results['foo.mean'], [4.4, 5.5, 6.6])
        self.assertAlmostEqualVector(results['foo.median'], [4.4, 5.5, 6.6])
        self.assertAlmostEqualVector(results['foo.min'], [1.1, 2.2, 3.3])
        self.assertAlmostEqualVector(results['foo.max'], [7.7, 8.8, 9.9])
        self.assertAlmostEqualVector(results['foo.var'], [7.26]*3)
        self.assertAlmostEqualVector(results['foo.dmean'], [3.3]*3)
        self.assertAlmostEqualVector(results['foo.dvar'], [0]*3)
        self.assertAlmostEqualVector(results['foo.dmean2'], [0]*3, precision=1e-6)
        self.assertAlmostEqualVector(results['foo.dvar2'], [0]*3, precision=1e-6)
  
        # test median for even number of frames
        p.add('foo', [10.0, 10.0, 10.0])
        results = PoolAggregator(defaultStats=defaultStats)(p)
        self.assertAlmostEqualVector(results['foo.median'], [6.05, 7.15, 8.25])

        # test cov and icov
        p = Pool({ 'foo': [[32.3, 43.21, 4.3],
                           [44.0, 5.12, 3.21],
                           [45.12, 415.0, 89.4],
                           [112.15, 45.0, 1.0023]]
                   })

        results = PoolAggregator(defaultStats=['cov', 'icov'])(p)

        self.assertAlmostEqualMatrix(results['foo.cov'],
                [[ 1317.99694824, -1430.0489502, -430.359405518],
                 [ -1430.0489502, 37181.1601563,  8301.80175781],
                 [-430.359405518, 8301.80175781,  1875.15136719]], precision = 1e-4)

        self.assertAlmostEqualMatrix(results['foo.icov'],
                [[ 0.00144919625018, -0.00161361286882, 0.0074764424935],
                 [-0.00161361729261,  0.00413948250934, -0.018696943298],
                 [ 0.00747651979327,  -0.0186969414353, 0.0850256085396]], precision = 1e-4)


    def testStringMatrixAggregation(self):
        p = Pool({ 'foo': [['qt', 'is', 'sweet'],
                           ['peanut', 'butter', 'jelly time'],
                           ['yo no', 'hablo', 'espanol']]
                   })

        results = PoolAggregator()(p)

        self.assertEqualVector(results.descriptorNames(), ['foo'])
        self.assertEqualMatrix(results['foo'], [['qt', 'is', 'sweet'],
                                                ['peanut', 'butter', 'jelly time'],
                                                ['yo no', 'hablo', 'espanol']])


    def testUnequalVectorLengths(self):
        p = Pool({ 'foo': [[548.4, 489.44, 45787.0],
                           [45.1, 78.0]]
                   })

        results = PoolAggregator()(p)

        # should result in no aggregation of the 'foo' descriptor
        self.assertEqualVector(results.descriptorNames(), [])


    # this test is similar to the one above, but the failure mode (if there
    # ever is one) might be different for empty vectors
    def testEmptyVectorInMatrix(self):
        p = Pool({ 'foo': [
                           [ 4.489,   22.0,          7.0],
                           [],
                           [89.153, .134, 10.1544564]
                          ]
                          })

        results = PoolAggregator()(p)

        # should result in no aggregation of the 'foo' descriptor
        self.assertEqualVector(results.descriptorNames(), [])


    # I would add a test that sees what the PoolAggregator does if there is
    # only one empty vector for a descriptor, but the python implementation
    # of Pool makes that impossible (see: PyPool.add)


    def testUnsupportedDefaultStatistic(self):
        defaultStats = ['min', 'max', 'hotness', 'var']

        self.assertConfigureFails(PoolAggregator(), {'defaultStats':defaultStats})


    def testUnsupportedExceptionStatistic(self):
        exceptions = {'weather': ['min', 'var'],
                      'pain': ['level']}

        self.assertConfigureFails(PoolAggregator(), {'exceptions': exceptions})


    def atestCopyStatistic(self):
        p = Pool({ 'foo': [3,6,3,2,39],
                   'bar.ff.wfew.f.gr.g.re.gr.e.gregreg.re.gr.eg.re': [[324, 5, 54], [543, 234, 57]]
                 })

        results = PoolAggregator(defaultStats=['copy'])(p)

        self.assertAlmostEqualVector(results['foo'], p['foo'])
        self.assertAlmostEqualMatrix(results['bar.ff.wfew.f.gr.g.re.gr.e.gregreg.re.gr.eg.re'],
                                           p['bar.ff.wfew.f.gr.g.re.gr.e.gregreg.re.gr.eg.re'])
        self.assertEqualVector(results.descriptorNames(),
                               ['foo', 'bar.ff.wfew.f.gr.g.re.gr.e.gregreg.re.gr.eg.re'])


    def testBadParamCopy(self):
        self.assertConfigureFails(PoolAggregator(), {'defaultStats': ['copy', 'min', 'max']})
        self.assertConfigureFails(PoolAggregator(), {'exceptions': {'exceptions': ['min', 'copy', 'var']}})


    def testValueStatistic(self):
        p = Pool( {'foo': [1,2,3,4,5]} )

        results = PoolAggregator(defaultStats=['max', 'value'])(p)

        self.assertTrue(results.containsKey('foo.max'))
        self.assertTrue(results.containsKey('foo.value'))
        self.assertAlmostEqual(results['foo.max'], 5)
        self.assertAlmostEqualVector(results['foo.value'], [1,2,3,4,5])

    def testArray2DAggregation(self):
        p = Pool()
        p.add('foo.bar', array([[0,1],[2,3]]))
        p.add('foo.bar', array([[1,2],[3,4]]))
        p.add('foo.bar', array([[2,3],[4,5]]))


        defaultStats = ['mean', 'min', 'max', 'var', 'dmean', 'dvar', 'dmean2', 'dvar2']
        results = PoolAggregator(defaultStats=defaultStats)(p)

        #mat is [[[0,1],[2,3]],[[1,2],[3,4]],[[2,3],[4,5]]]
        self.assertEqualMatrix(results['foo.bar.min'],[[0,1], [2,3]])
        self.assertEqualMatrix(results['foo.bar.max'],[[2,3], [4,5]])
        self.assertEqualMatrix(results['foo.bar.mean'],[[1,2], [3,4]])
        self.assertAlmostEqualMatrix(results['foo.bar.var'],[[2./3.,2./3.], [2./3., 2./3.]])
        self.assertEqualMatrix(results['foo.bar.dmean'],[[1,1], [1,1]])
        self.assertEqualMatrix(results['foo.bar.dvar'],[[0,0], [0,0]])
        self.assertEqualMatrix(results['foo.bar.dmean2'],[[0,0], [0,0]])
        self.assertEqualMatrix(results['foo.bar.dvar2'], [[0,0],[0,0]])


suite = allTests(TestPoolAggregator)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
