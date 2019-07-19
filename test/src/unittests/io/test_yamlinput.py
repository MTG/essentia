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



from essentia_test import *
import os


def writeFile(str, filename):
    resultF = open(filename, 'w')
    resultF.write(str)
    resultF.close()


class TestYamlInput(TestCase):

    def testSingleKeySingleReal(self):
        testFile = 'foo: 1.0'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertAlmostEqual(p['foo'], 1.0)

        testJson = '{ "foo": 1.0 }'
        writeFile(testJson, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertAlmostEqual(p['foo'], 1.0)

    def testSingleKeySingleString(self):
        testFile = 'foo: \"herro\"'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqual(p['foo'], 'herro')

        testJson = '{ \"foo\": \"herro\" }'
        writeFile(testJson, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertEqual(p['foo'], 'herro')

    def testSingleKeyVecStrSingle(self):
        testFile = 'foo: ["bar"]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], ['bar'])

        testJson = '{ "foo": ["bar"] }'
        writeFile(testJson, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], ['bar'])

    def testSingleKeyVecStr(self):
        testFile = 'foo: ["bar", "foo", "moo", "shu"]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], ['bar', 'foo', 'moo', 'shu'])

        testJson = '{ "foo": ["bar", "foo", "moo", "shu"] }'
        writeFile(testJson, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], ['bar', 'foo', 'moo', 'shu'])


    def testSingleKeyVecVecStr(self):
        testFile = 'foo: [["bar", "bar2", "bar3"], ["foo", "foo1"], ["moo"], ["shu", "shu1", "shu2", "shu3"]]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')
        self.assertEqualVector(p['foo'], [['bar', 'bar2', 'bar3'],
                                                ['foo', 'foo1'],
                                                ['moo'],
                                                ['shu', 'shu1', 'shu2', 'shu3']])

        testFile = '{ "foo": [["bar", "bar2", "bar3"], ["foo", "foo1"], ["moo"], ["shu", "shu1", "shu2", "shu3"]] }'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')
        self.assertEqualVector(p['foo'], [['bar', 'bar2', 'bar3'],
                                                ['foo', 'foo1'],
                                                ['moo'],
                                                ['shu', 'shu1', 'shu2', 'shu3']])

    def testSingleKey(self):
        testFile = 'foo: [1.0]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertAlmostEqualVector(p['foo'], [1.0])

        testFile = '{ "foo": [1.0] }'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertAlmostEqualVector(p['foo'], [1.0])

    def testMultiKey(self):
        testFile = '''
foo: ["I", "am", "Batman"]

bar: [32, 654, 3234, 324]

cat: [["eyow", "random", "fjrwoi"], ["hrue", "fhrehfkjhf", "jksnkvdsvsvdsvsfdsfew fw rfewfe"], ["twy7848723#4 34928 98f////fewfewfew"]]

array: [[123, 156.156, 123, 78], [-78714.123], [78452]]
'''
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        descNames = sorted(p.descriptorNames())
        self.assertEqualVector(descNames, ['array', 'bar', 'cat', 'foo'])
        self.assertEqualVector(p['foo'], ['I', 'am', 'Batman'])
        self.assertAlmostEqualVector(p['bar'], [32, 654, 3234, 324])
        self.assertEqualMatrix(p['cat'], [['eyow', 'random', 'fjrwoi'], ['hrue', 'fhrehfkjhf', 'jksnkvdsvsvdsvsfdsfew fw rfewfe'], ['twy7848723#4 34928 98f////fewfewfew']])
        self.assertAlmostEqualMatrix(p['array'], [[123, 156.156, 123, 78], [-78714.123], [78452]])

        testFile = '''
{ 
"foo": ["I", "am", "Batman"],

"bar": [32, 654, 3234, 324],

"cat": [["eyow", "random", "fjrwoi"], ["hrue", "fhrehfkjhf", "jksnkvdsvsvdsvsfdsfew fw rfewfe"], ["twy7848723#4 34928 98f////fewfewfew"]],

"array": [[123, 156.156, 123, 78], [-78714.123], [78452]]
}'''
        writeFile(testFile, 'testfile')
        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        descNames = sorted(p.descriptorNames())
        self.assertEqualVector(descNames, ['array', 'bar', 'cat', 'foo'])
        self.assertEqualVector(p['foo'], ['I', 'am', 'Batman'])
        self.assertAlmostEqualVector(p['bar'], [32, 654, 3234, 324])
        self.assertEqualMatrix(p['cat'], [['eyow', 'random', 'fjrwoi'], ['hrue', 'fhrehfkjhf', 'jksnkvdsvsvdsvsfdsfew fw rfewfe'], ['twy7848723#4 34928 98f////fewfewfew']])
        self.assertAlmostEqualMatrix(p['array'], [[123, 156.156, 123, 78], [-78714.123], [78452]])

    def testMultiScopedKeys(self):
        testFile= '''
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
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        descNames = sorted(p.descriptorNames())
        self.assertEqualVector(descNames, ['cat.bat.hat', 'cat.sub', 'foo.bar', 'foo.second', 'john.joe.jane', 'zat'])
        self.assertEqualVector(p['foo.bar'], ['hello', 'world'])
        self.assertAlmostEqualVector(p['foo.second'], [1, 2, 3, 4, 5])
        self.assertAlmostEqualMatrix(p['cat.bat.hat'], [[0, 9, -8, -7, -6], [-5, 4, 3, 2]])
        self.assertEqualMatrix(p['cat.sub'], [['lets'], ['see'], ['if'], ['it', 'breaks']])
        self.assertEqualMatrix(p['zat'], ['foo'])
        self.assertAlmostEqual(p['john.joe.jane'], 3)

        testFile= '''
{
"foo": 
    {
    "bar": ["hello", "world"],
    "second": [1, 2, 3, 4, 5]
    },
"cat": 
    {
    "bat": 
        {
        "hat": [[0, 9, -8, -7, -6], [-5, 4, 3, 2]]
        },
    "sub": [["lets"], ["see"], ["if"], ["it", "breaks"]]
    },
"zat": ["foo"],
"john":
    {
    "joe":
        {
        "jane": 3
        }
    }
}
'''
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        descNames = sorted(p.descriptorNames())
        self.assertEqualVector(descNames, ['cat.bat.hat', 'cat.sub', 'foo.bar', 'foo.second', 'john.joe.jane', 'zat'])
        self.assertEqualVector(p['foo.bar'], ['hello', 'world'])
        self.assertAlmostEqualVector(p['foo.second'], [1, 2, 3, 4, 5])
        self.assertAlmostEqualMatrix(p['cat.bat.hat'], [[0, 9, -8, -7, -6], [-5, 4, 3, 2]])
        self.assertEqualMatrix(p['cat.sub'], [['lets'], ['see'], ['if'], ['it', 'breaks']])
        self.assertEqualMatrix(p['zat'], ['foo'])
        self.assertAlmostEqual(p['john.joe.jane'], 3)

    def testSingleQuotes(self):
        testFile = 'foo: [\'bar\', \'shu\']'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], ['bar', 'shu'])

        # NOTE: in the case of json        
        # jsonconvert will teat single-quoted text as a number and it does not apply check for number correctenss
        # therefore, exception will not be thrown

        #testFile = '{ "foo": [\'bar\', \'shu\'] }'
        #writeFile(testFile, 'testfile')
        #self.assertRaises(RuntimeError, lambda: YamlInput(filename='testfile', format='json')())
        #os.remove('testfile')

    def testStereoSample(self):
        testFile = 'foo: [{left: 3, right: 6}, {left: 4, right: 7}]'
        writeFile(testFile, 'testfile')

        try:
            p = YamlInput(filename='testfile')()
        finally:
            os.remove('testfile')

        self.assertEqual(len(p['foo']), 2)
        self.assertEqual(p['foo'][0][0], 3)
        self.assertEqual(p['foo'][0][1], 6)
        self.assertEqual(p['foo'][1][0], 4)
        self.assertEqual(p['foo'][1][1], 7)

        testFile = '{ "foo": [{"left": 3, "right": 6}, {"left": 4, "right": 7}] }'
        writeFile(testFile, 'testfile')
        
        # TODO dict elements in lists  are currently not supported for json
        self.assertRaises(RuntimeError, lambda: YamlInput(filename='testfile', format='json')())
        os.remove('testfile')
        
    def testInvalidFile(self):
        self.assertRaises(RuntimeError, lambda: YamlInput(filename='blablabla.yaml')())
        self.assertRaises(RuntimeError, lambda: YamlInput(filename='blablabla.yaml', format='json')())

    def testSequenceWhichContainsEmptySequences(self):
        # yaml
        testFile = 'foo: [[], []]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqualMatrix(p['foo'], [[], []])

        # to make sure that they are vectors of vectors Reals, try adding Reals
        # and strings
        self.assertRaises(KeyError, lambda: p.add('foo', ['foo', 'bar']))

        p.add('foo', [4])
        self.assertEqualMatrix(p['foo'], [[], [], [4]])

        # json 
        testFile = '{ "foo": [[], []] }'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertEqualMatrix(p['foo'], [[], []])

        # to make sure that they are vectors of vectors Reals, try adding Reals
        # and strings
        self.assertRaises(KeyError, lambda: p.add('foo', ['foo', 'bar']))

        p.add('foo', [4])
        self.assertEqualMatrix(p['foo'], [[], [], [4]])

    def testSequenceWhichContainsEmptySequenceButCanDetermineType(self):
        # yaml
        testFile = 'foo: [[], [\"wassup\"], []]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], [[], ['wassup'], []])
        
        # json
        testFile = '{ "foo": [[], [\"wassup\"], []] }'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertEqualVector(p['foo'], [[], ['wassup'], []])

    def testMixedTypes(self):
        # yaml
        testFile = 'foo: [3.4, \"wassup\"]'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile'))
        os.remove('testfile')

        # json
        testFile = '{ "foo": [3.4, \"wassup\"] }'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile', format='json'))
        os.remove('testfile')

    def testVectorMatrix(self):
        # yaml
        testFile = 'foo: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile')()
        os.remove('testfile')

        self.assertEqualMatrix(p['foo'][0], [[1,2],[3,4]])
        self.assertEqualMatrix(p['foo'][1], [[5,6],[7,8]])

        # json
        testFile = '{ "foo": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] }'
        writeFile(testFile, 'testfile')

        p = YamlInput(filename='testfile', format='json')()
        os.remove('testfile')

        self.assertEqualMatrix(p['foo'][0], [[1,2],[3,4]])
        self.assertEqualMatrix(p['foo'][1], [[5,6],[7,8]])

    def testVectorMatrixMixed(self):
        # yaml
        testFile = 'foo: [[[1, 2], [3, 4]], 3]'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile'))
        os.remove('testfile')

        # json
        testFile = '{ "foo": [[[1, 2], [3, 4]], 3] }'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile', format='json'))
        os.remove('testfile')

    def testVectorMatrixString(self):
        # yaml
        testFile = 'foo: [[["foo", "bar"], ["shu", "moo"]]]'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile'))
        os.remove('testfile')

        # json
        testFile = '{ "foo": [[["foo", "bar"], ["shu", "moo"]]] }'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile', format='json'))
        os.remove('testfile')

    def testVectorMatrixNotRectangular(self):
        # yaml
        testFile = 'foo: [[[1, 2], [3, 4]], [[1,2], []] ]'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile'))
        os.remove('testfile')
        
        # json
        testFile = '{ "foo": [[[1, 2], [3, 4]], [[1,2], []] ] }'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile', format='json'))
        os.remove('testfile')

    def testVectorMatrixEmpty(self):
        # yaml
        testFile = 'foo: [[[], []]]'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile'))
        os.remove('testfile')
        
        # json
        testFile = '{ foo: [[[], []]] }'
        writeFile(testFile, 'testfile')

        self.assertComputeFails(YamlInput(filename='testfile', format='json'))
        os.remove('testfile')
    
    def testJsonEscapedStrings(self):
        p = Pool()
        p.add('vector_string', 'string_1\n\r " \ /')
        p.add('vector_string', 'string_2\n\r " \ /')
        p.add('vector_string', 'string_3\n\r " \ /')
        p.set('string', 'string\n\r " \ /')

        YamlOutput(filename='testfile', format='json')(p)
        p_loaded = YamlInput(filename='testfile', format='json')()

        self.assertEqual(p['vector_string'], ['string_1\n\r " \ /', 
                                              'string_2\n\r " \ /', 
                                              'string_3\n\r " \ /'])
        self.assertEqual(p['string'], 'string\n\r " \ /')

        os.remove('testfile')


    def testEmptyFile(self):
        testFile = ''
        writeFile(testFile, 'testfile')

        self.assertRaises(RuntimeError, lambda: YamlInput(filename='testfile')())
        self.assertRaises(RuntimeError, lambda: YamlInput(filename='testfile', format='json')())
        os.remove('testfile')


suite = allTests(TestYamlInput)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
