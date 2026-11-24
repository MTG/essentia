#!/usr/bin/env python

# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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


# Regression tests for https://github.com/MTG/essentia/issues/405 (memory leak
# in streaming mode in python): dropping the Python proxies of a streaming
# network must release the C++ side of the network as well -- the connection
# ring buffers, the algorithms themselves, and the internal PoolStorage
# algorithms created when connecting a source to a Pool.

import gc
import resource
import sys

from essentia_test import *
from essentia.streaming import _StreamConnector, disconnectNetwork
import essentia.streaming as es
import numpy


# ru_maxrss is expressed in kilobytes on Linux and in bytes on macOS/BSD
_MAXRSS_TO_MB = 1024.0 if sys.platform.startswith('linux') else (1024.0 * 1024.0)


def _highWaterMB():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _MAXRSS_TO_MB


class TestNetworkTeardown(TestCase):

    def _buildFanoutNetwork(self, pool, nbranches=8):
        # A fan-out network with algorithms more than one connection away from
        # the generator. The streaming Scale allocates a
        # BufferUsage::forLargeAudioStream output buffer (4MB), so every
        # branch that outlives its Python proxies is clearly visible in RSS.
        gen = VectorInput(essentia.array(numpy.zeros(1024, dtype='float32')))
        head = es.Scale(factor=1.0)
        gen.data >> head.signal

        branches = []
        for i in range(nbranches):
            scale = es.Scale(factor=1.0)
            head.signal >> scale.signal
            scale.signal >> (pool, 'branch_%d' % i)
            branches.append(scale)

        return gen, head, branches

    def testRepeatedBuildAndDropDoesNotGrowRss(self):
        # each dropped round used to retain ~4MB per branch (the C++ side of
        # everything further than one connection from the generator was never
        # deleted), so 10 rounds x 8 branches would grow the high-water mark
        # by roughly 300MB. With the leak fixed the growth after the first
        # round is zero; the assertion budget is kept deliberately generous so
        # that unrelated allocator noise cannot make this test flaky.
        rounds = 10
        baseline = None
        for i in range(rounds):
            pool = Pool()
            gen, head, branches = self._buildFanoutNetwork(pool)
            del gen, head, branches, pool
            # the proxies live in reference cycles, so they are reclaimed by
            # the cyclic garbage collector, not by reference counting
            gc.collect()
            # take the baseline after the second round: the first rounds can
            # bump the high-water mark once as the allocator warms up, and the
            # high-water mark never comes back down
            if i < 2:
                baseline = _highWaterMB()

        growth = _highWaterMB() - baseline
        self.assertTrue(
            growth < 40.0,
            'high-water RSS grew by %.1f MB across %d build+drop rounds' % (growth, rounds))

    def testRepeatedRunAndDropDoesNotGrowRss(self):
        rounds = 10
        baseline = None
        expected = None
        for i in range(rounds):
            pool = Pool()
            gen, head, branches = self._buildFanoutNetwork(pool)
            run(gen)
            result = pool['branch_0']
            # teardown must not perturb the computed results
            if expected is None:
                expected = result
            else:
                self.assertEqualVector(result, expected)
            del gen, head, branches, pool
            gc.collect()
            if i < 2:
                baseline = _highWaterMB()

        growth = _highWaterMB() - baseline
        self.assertTrue(
            growth < 40.0,
            'high-water RSS grew by %.1f MB across %d run+drop rounds' % (growth, rounds))

    def testRepeatedAlgorithmCreationDoesNotGrowRss(self):
        # the original reproducer of MTG/essentia#405: unconnected streaming
        # algorithms created in a loop
        rounds = 300
        baseline = None
        for i in range(rounds):
            algo = es.MFCC()
            del algo
            gc.collect()
            if i < 2:
                baseline = _highWaterMB()

        growth = _highWaterMB() - baseline
        self.assertTrue(
            growth < 20.0,
            'high-water RSS grew by %.1f MB across %d algorithm create+drop rounds' % (growth, rounds))

    def testDisconnectNetworkSeversAllConnections(self):
        pool = Pool()
        gen, head, branches = self._buildFanoutNetwork(pool)

        # passing any single algorithm of the network is enough
        disconnectNetwork(branches[0])

        for algo in [gen, head] + branches:
            for connector, targets in algo.connections.items():
                self.assertEqualVector(targets, [])
            self.assertEqualVector(getattr(algo, '_upstream', []), [])

    def testDisconnectNetworkIsIdempotentAndTolerant(self):
        pool = Pool()
        gen, head, branches = self._buildFanoutNetwork(pool)

        # a partially severed network must not raise
        head.signal.disconnect(branches[0].signal)
        disconnectNetwork(gen)
        disconnectNetwork(gen, head, None, *branches)

        # unconnected and None algorithms must not raise either
        disconnectNetwork(None)
        disconnectNetwork(es.Scale())

    def testNetworkStaysRunnableWhileAnyProxyIsReferenced(self):
        # the sink side keeps the source side alive: dropping the explicit
        # references to the upstream algorithms must not break the network as
        # long as part of it is still referenced
        pool = Pool()
        gen, head, branches = self._buildFanoutNetwork(pool)
        del head
        gc.collect()

        run(gen)
        self.assertTrue(len(pool.descriptorNames()) == len(branches))

    def testResultsUnchangedAfterExplicitTeardown(self):
        data = essentia.array(numpy.arange(1024, dtype='float32'))

        def computeOnce():
            pool = Pool()
            gen = VectorInput(data)
            scale = es.Scale(factor=0.5)
            gen.data >> scale.signal
            scale.signal >> (pool, 'scaled')
            run(gen)
            result = pool['scaled']
            disconnectNetwork(gen)
            del gen, scale, pool
            gc.collect()
            return result

        first = computeOnce()
        second = computeOnce()
        self.assertEqualVector(second, first)


suite = allTests(TestNetworkTeardown)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
