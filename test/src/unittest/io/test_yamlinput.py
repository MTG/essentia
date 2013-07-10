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
import os


def writeYaml(str, filename):
    resultF = open(filename, 'w')
    resultF.write(str)
    resultF.close()


class TestYamlInput(TestCase):

    def testSingleKeySingleReal(self):
        testYaml = 'foo: 1.0'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertAlmostEqual(p['foo'], 1.0)

    def testSingleKeySingleString(self):
        testYaml = 'foo: \"herro\"'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqual(p['foo'], 'herro')

    def testSingleKeyVecStrSingle(self):
        testYaml = 'foo: ["bar"]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqualVector(p['foo'], ['bar'])

    def testSingleKeyVecStr(self):
        testYaml = 'foo: ["bar", "foo", "moo", "shu"]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqualVector(p['foo'], ['bar', 'foo', 'moo', 'shu'])

    def testSingleKeyVecVecStr(self):
        testYaml = 'foo: [["bar", "bar2", "bar3"], ["foo", "foo1"], ["moo"], ["shu", "shu1", "shu2", "shu3"]]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')
        self.assertEqualVector(p['foo'], [['bar', 'bar2', 'bar3'],
                                                ['foo', 'foo1'],
                                                ['moo'],
                                                ['shu', 'shu1', 'shu2', 'shu3']])

    def testSingleKey(self):
        testYaml = 'foo: [1.0]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertAlmostEqualVector(p['foo'], [1.0])

    def testMultiKey(self):
        testYaml = '''
foo: ["I", "am", "Batman"]

bar: [32, 654, 3234, 324]

cat: [["eyow", "random", "fjrwoi"], ["hrue", "fhrehfkjhf", "jksnkvdsvsvdsvsfdsfew fw rfewfe"], ["twy7848723#4 34928 98f////fewfewfew"]]

array: [[123, 156.156, 123, 78], [-78714.123], [78452]]
'''
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        descNames = sorted(p.descriptorNames())
        self.assertEqualVector(descNames, ['array', 'bar', 'cat', 'foo'])
        self.assertEqualVector(p['foo'], ['I', 'am', 'Batman'])
        self.assertAlmostEqualVector(p['bar'], [32, 654, 3234, 324])
        self.assertEqualMatrix(p['cat'], [['eyow', 'random', 'fjrwoi'], ['hrue', 'fhrehfkjhf', 'jksnkvdsvsvdsvsfdsfew fw rfewfe'], ['twy7848723#4 34928 98f////fewfewfew']])
        self.assertAlmostEqualMatrix(p['array'], [[123, 156.156, 123, 78], [-78714.123], [78452]])


    def testMultiScopedKeys(self):
        testYaml= '''
foo:
    bar: ["hello", "world"]
    second: [1, 2, 3, 4, 5]

cat:
    bat:
        hat: [[0, 9, -8, -7, -6], [-5, 4, 3, 2]]
    sub: [["lets"], ["see"], ["if"], ["it", "breaks"]]

zat: ["foo"]

john:
    joe:
        jane: 3
'''
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        descNames = sorted(p.descriptorNames())
        self.assertEqualVector(descNames, ['cat.bat.hat', 'cat.sub', 'foo.bar', 'foo.second', 'john.joe.jane', 'zat'])
        self.assertEqualVector(p['foo.bar'], ['hello', 'world'])
        self.assertAlmostEqualVector(p['foo.second'], [1, 2, 3, 4, 5])
        self.assertAlmostEqualMatrix(p['cat.bat.hat'], [[0, 9, -8, -7, -6], [-5, 4, 3, 2]])
        self.assertEqualMatrix(p['cat.sub'], [['lets'], ['see'], ['if'], ['it', 'breaks']])
        self.assertEqualMatrix(p['zat'], ['foo'])
        self.assertAlmostEqual(p['john.joe.jane'], 3)

    def testSingleQuotes(self):
        testYaml = 'foo: [\'bar\', \'shu\']'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqualVector(p['foo'], ['bar', 'shu'])

    def testStereoSample(self):
        testYaml = 'foo: [{left: 3, right: 6}, {left: 4, right: 7}]'

        writeYaml(testYaml, 'test.yaml')

        try:
            p = YamlInput(filename='test.yaml')()
        finally:
            os.remove('test.yaml')

        self.assertEqual(len(p['foo']), 2)
        self.assertEqual(p['foo'][0][0], 3)
        self.assertEqual(p['foo'][0][1], 6)
        self.assertEqual(p['foo'][1][0], 4)
        self.assertEqual(p['foo'][1][1], 7)

    def testInvalidFile(self):
        self.assertRaises(RuntimeError, lambda: YamlInput(filename='blablabla.yaml')())

    def testSequenceWhichContainsEmptySequences(self):
        testYaml = 'foo: [[], []]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqualVector(p['foo'], [[], []])

        # to make sure that they are vectors of vectors Reals, try adding Reals
        # and strings
        self.assertRaises(KeyError, lambda: p.add('foo', ['foo', 'bar']))

        p.add('foo', [4])
        self.assertEqualVector(p['foo'], [[], [], [4]])


    def testSequenceWhichContainsEmptySequenceButCanDetermineType(self):
        testYaml = 'foo: [[], [\"wassup\"], []]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqualVector(p['foo'], [[], ['wassup'], []])

    def testMixedTypes(self):
        testYaml = 'foo: [3.4, \"wassup\"]'
        writeYaml(testYaml, 'test.yaml')

        self.assertComputeFails(YamlInput(filename='test.yaml'))
        os.remove('test.yaml')

    def testVectorMatrix(self):
        testYaml = 'foo: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]'
        writeYaml(testYaml, 'test.yaml')

        p = YamlInput(filename='test.yaml')()
        os.remove('test.yaml')

        self.assertEqualMatrix(p['foo'][0], [[1,2],[3,4]])
        self.assertEqualMatrix(p['foo'][1], [[5,6],[7,8]])


    def testVectorMatrixMixed(self):
        testYaml = 'foo: [[[1, 2], [3, 4]], 3]'
        writeYaml(testYaml, 'test.yaml')

        self.assertComputeFails(YamlInput(filename='test.yaml'))
        os.remove('test.yaml')

    def testVectorMatrixString(self):
        testYaml = 'foo: [[["foo", "bar"], ["shu", "moo"]]]'
        writeYaml(testYaml, 'test.yaml')

        self.assertComputeFails(YamlInput(filename='test.yaml'))
        os.remove('test.yaml')

    def testVectorMatrixNotRectangular(self):
        testYaml = 'foo: [[[1, 2], [3, 4]], [[1,2], []] ]'
        writeYaml(testYaml, 'test.yaml')

        self.assertComputeFails(YamlInput(filename='test.yaml'))
        os.remove('test.yaml')

    def testVectorMatrixEmpty(self):
        testYaml = 'foo: [[[], []]]'
        writeYaml(testYaml, 'test.yaml')

        self.assertComputeFails(YamlInput(filename='test.yaml'))
        os.remove('test.yaml')



suite = allTests(TestYamlInput)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
