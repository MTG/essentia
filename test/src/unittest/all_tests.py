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



from os.path import join,  sep
import os
import sys
import unittest
import glob
import essentia
import essentia.streaming

# we don't want to get too chatty when running all the tests
essentia.log.info = False
#essentia.log.debug += essentia.EAll
#essentia.log.debug -= essentia.EConnectors

# chdir into the dir of this file so paths work out right
tests_dir = os.path.dirname(__file__)
if tests_dir:
    os.chdir(tests_dir)

# import the test from the subdirectories which filename match the pattern 'test_*.py'
listAllTests = [ filename.split(sep+'test_') for filename in glob.glob(join('*', 'test_*.py')) ]
for testfile in listAllTests:
    testfile[1] = testfile[1][:-3]



def importTest(fullname, strategy = 'import'):
    '''Imports or reloads test given its fullname.'''
    folder, name = fullname
    if strategy == 'import':
        cmd = 'import %s.test_%s; setattr(sys.modules[__name__], \'%s\', %s.test_%s.suite)' % (folder, name, name, folder, name)
    elif strategy == 'reload':
        cmd1 = 'reload(sys.modules[\'%s.test_%s\']); ' % (folder, name)
        cmd2 = 'setattr(sys.modules[__name__], \'%s\', sys.modules[\'%s.test_%s\'].suite)' % (name, folder, name)

        cmd = cmd1 + cmd2
    else:
        raise ValueError('When importing a test, only strategies allowed are \'import\' and \'reload\'')

    exec(cmd)



def getTests(names=None, exclude=None, strategy='import'):
    allNames = [ name for _, name in listAllTests ]
    names = names or allNames
    tests = [ (folder, name) for folder, name in listAllTests
              if name in names and name not in exclude ]

    for name in names:
        if name not in allNames:
            print 'WARNING: did not find test', name
    for name in (exclude or []):
        if name not in allNames:
            print 'WARNING: did not find test to exclude', name

    print 'Running tests:', sorted(name for _, name in tests)

    if not tests:
        raise RuntimeError('No test to execute!')

    for test in tests:
        importTest(test, strategy)

    testObjectsList = [ getattr(sys.modules[__name__], testName) for folder, testName in tests ]

    return unittest.TestSuite(testObjectsList)



def traceCompute(algo, *args, **kwargs):
    print 'computing algo', algo.name()
    return algo.normalCompute(*args, **kwargs)


def computeResetCompute(algo, *args, **kwargs):
    # do skip certain algos, otherwise we'd enter in an infinite loop!!!
    audioLoaders = [ 'MonoLoader', 'EqloudLoader', 'EasyLoader', 'AudioLoader' ]
    filters = [ 'IIR', 'DCRemoval', 'HighPass', 'LowPass', 'BandPass', 'AllPass',
                'BandReject', 'EqualLoudness', 'MovingAverage' ]
    special = [ 'FrameCutter', 'TempoScaleBands', 'TempoTap', 'TempoTapTicks',
                'Panning','OnsetDetection', 'MonoWriter', 'Flux', 'StartStopSilence' ]

    if algo.name() in audioLoaders + filters + special:
        return algo.normalCompute(*args, **kwargs)
    else:
        algo.normalCompute(*args, **kwargs)
        algo.reset()
        return algo.normalCompute(*args, **kwargs)


def computeDecorator(newCompute):
    def algodecorator(algo):
        algo.normalCompute = algo.compute
        algo.compute = newCompute
        algo.__call__ = newCompute
        algo.hasDoubleCompute = True
        return algo

    return algodecorator

# recursive helper function that finds outputs connected to pools and calls func
def mapPools(algo, func):
    # make a copy first, because func might modify the connections in the for
    # loops
    connections = dict(algo.connections)

    for output, inputs in connections.iteritems():
        ins = list(inputs)
        for input in ins:
            # TODO: assuming input is a tuple of pool and descriptor name

            if isinstance(input, tuple):
                func(algo, output, input)

            elif isinstance(input, essentia.streaming._StreamConnector):
                mapPools(input.input_algo, func)

            #else ignore nowhere connections



# For this to work for networks that are connected to a pool, we need to conduct
# the first run of the network with all pools replaced by dummy pools. The
# second run will run with the network connected to the original pools. This
# method is required to avoid doubling of the data in the pools due to the fact
# that we run the network twice.
def runResetRun(gen, *args, **kwargs):
    # 0. Find networks which contain algorithms who do not play nice with our
    #    little trick. In particular, we have a test for multiplexer that runs
    #    multiple generators...
    def isValid(algo):
        if isinstance(algo, essentia.streaming.VectorInput) and not algo.connections.values()[0]:
            # non-connected VectorInput, we don't want to get too fancy here...
            return False
        if algo.name() == 'Multiplexer':
            return False
        for output, inputs in algo.connections.iteritems():
            for inp in inputs:
                if isinstance(inp, essentia.streaming._StreamConnector) and not isValid(inp.input_algo):
                    return False
        return True

    if not isValid(gen):
        print 'Network is not capable of doing the run/reset/run trick, doing it the normal way...'
        essentia.run(gen)
        return


    # 1. Find all the outputs in the network that are connected to pools--aka
    #    pool feeders and for each pool feeder, disconnect the given pool,
    #    store it, and connect a dummy pool in its place
    def useDummy(algo, output, input):
        if not hasattr(output, 'originalPools'):
            output.originalPools = []
            output.dummyPools = []

        # disconnect original
        output.originalPools.append(input)
        output.disconnect(input)

        # connect dummy
        dummy = essentia.Pool()
        output.dummyPools.append((dummy, input[1]))
        output >> output.dummyPools[-1]

    mapPools(gen, useDummy)

    # 2. Run the network
    essentia.run(gen)

    # 3. Reset the network
    essentia.reset(gen)

    # 4. For each pool feeder, disconnect the dummy pool and reconnect the
    #    original pool
    def useOriginal(algo, output, input):
        # disconnect dummy
        output.disconnect(input)
        # the dummy pools and the original pools should have the same index

        idx = output.dummyPools.index(input)
        output.dummyPools.remove(input)

        # connect original
        output >> output.originalPools[idx]

        # don't need these anymore
        if len(output.dummyPools) == 0:
            del output.dummyPools
            del output.originalPools

    mapPools(gen, useOriginal)

    # 5. Run the network for the second and final time
    return essentia.run(gen)



def runTests(tests):
    result = unittest.TextTestRunner(verbosity=2).run(tests)

    # return the number of failures and errors
    return len(result.errors) + len(result.failures)


if __name__ == '__main__':
    testList = [ t for t in sys.argv[1:] if t[0] != '-' ]
    testExclude = [ t[1:] for t in sys.argv[1:] if t[0] == '-' ]

    print 'TEST LIST', testList
    print 'TEST EXCLUDE', testExclude

    print 'Running tests normally'
    print '-'*70
    result1 = runTests(getTests(testList, exclude=testExclude))

    print '\n\nRunning tests with compute/reset/compute'
    print '-'*70

    setattr(sys.modules['essentia.common'], 'algoDecorator', computeDecorator(computeResetCompute))
    essentia.standard._reloadAlgorithms()
    essentia.standard._reloadAlgorithms('essentia_test')

    # modify runGenerator behavior
    setattr(sys.modules['essentia_test'], 'run', runResetRun)


    result2 = runTests(getTests(testList, exclude=testExclude, strategy='reload'))

    sys.exit(result1 + result2)
