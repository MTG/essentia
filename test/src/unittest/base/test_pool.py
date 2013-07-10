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

class TestPool(TestCase):

    def testIntToRealConversion(self):
        p = Pool()
        p.add('foo', 1)
        p.add('foo', 2)
        p.add('foo', 3)
        p.add('foo', 5)
        p.add('foo', 7)
        p.add('foo', 11)
        p.add('foo', 13)

        self.assertAlmostEqualVector(p['foo'], [1, 2, 3, 5, 7, 11, 13])

    # Make sure the basic use case works
    def testRealPoolSimple(self):
        expectedVal = 6.9

        p = Pool()
        p.add('foo.bar', expectedVal)

        self.assertAlmostEqualVector(p['foo.bar'], [expectedVal])

    # Make sure we can support having multiple values under one label
    def testRealPoolMultiple(self):
        expectedVal1 = 6.9
        expectedVal2 = 16.0

        p = Pool()
        p.add('foo.bar', expectedVal1)
        p.add('foo.bar', expectedVal2)

        self.assertAlmostEqualVector(p['foo.bar'], [expectedVal1, expectedVal2])

    # Make sure we can support having multiple labels in the same pool
    def testRealPoolMultipleLabels(self):
        expectedVal1 = 6.9
        expectedVal2 = 16.0

        p = Pool()
        p.add('foo.bar', expectedVal1)
        p.add('bar.foo', expectedVal2)

        self.assertAlmostEqualVector(p['foo.bar'], [expectedVal1])
        self.assertAlmostEqualVector(p['bar.foo'], [expectedVal2])

    def testRealVectorPoolSimple(self):
        expectedVec = [1.6, 0.9, 19.85]

        p = Pool()
        p.add('foo.bar', expectedVec)

        self.assertAlmostEqualMatrix(p['foo.bar'], [expectedVec])

    def testRealVectorPoolMultiple(self):
        expectedVec1 = [1.6, 0.9, 19.85]
        expectedVec2 = [-5.0, 0.0, 5.0]

        p = Pool()
        p.add('foo.bar', expectedVec1)
        p.add('foo.bar', expectedVec2)

        self.assertAlmostEqualMatrix(p['foo.bar'], [expectedVec1, expectedVec2])

    def testRealVectorPoolMultipleLabels(self):
        expectedVec1 = [1.6, 0.9, 19.85]
        expectedVec2 = [-5.0, 0.0, 5.0]

        p = Pool()
        p.add('foo.bar', expectedVec1)
        p.add('bar.foo', expectedVec2)

        self.assertAlmostEqualMatrix(p['foo.bar'], [expectedVec1])
        self.assertAlmostEqualMatrix(p['bar.foo'], [expectedVec2])

    # Test adding an empty vector
    def testVectorEmpty(self):
        p = Pool()
        p.add('foo.bar', [1])
        p.add('foo.bar', [])

        self.assertEqualMatrix(p['foo.bar'], [[1], []])

    # Make sure the lookup of a non-existant descriptorName fails
    def testMissingDescriptorName(self):
        p = Pool()
        p.add('foo.bar', 0.0)

        self.assertRaises(KeyError, p.__getitem__, 'bar.bar')
        self.assertTrue(not p.containsKey('bar.bar'))

    def testRemove(self):
        expectedVal = 123.456

        p = Pool()
        p.add('foo.rab', expectedVal)
        p.add('foo.bar', 0.0)
        p.add('foo.bar', 1111.1111)
        p.remove('foo.bar')

        self.assertRaises(KeyError, p.__getitem__, 'foo.bar')
        self.assertAlmostEqualVector(p['foo.rab'], [expectedVal])

    def testRemoveNamespace(self):

        p = Pool()
        p.add('real.one', 1.0)
        p.add('real.two', 2.0)
        p.add('real.three.a', 'a')
        p.add('real.three.b', 'b')

        p.removeNamespace('real.three')

        self.assertEqualVector(p.descriptorNames(), ['real.one', 'real.two'])
        self.assertEqualVector(p['real.one'], [1.0])
        self.assertEqualVector(p['real.two'], [2.0])

        p.removeNamespace('real')
        self.assertEqualVector(p.descriptorNames(), [])

    def testClear(self):

        p = Pool()
        p.add('real.one', 1.0)
        p.add('real.two', 2.0)
        p.add('real.three.a', 'a')
        p.add('real.three.b', 'b')

        p.clear()

        self.assertEqualVector(p.descriptorNames(), [])

  # String type tests

    def testStringPoolSimple(self):
        p = Pool()
        p.add('foo.bar', 'simple')

        self.assertEqualVector(p['foo.bar'], ['simple'])

    def testStringPoolMultiple(self):
        expectedVal1 = 'mul'
        expectedVal2 = 'tiple'

        p = Pool()
        p.add('foo.bar', expectedVal1)
        p.add('foo.bar', expectedVal2)

        self.assertEqualVector(p['foo.bar'], [expectedVal1, expectedVal2])

    def testStringPoolMultipleLabels(self):
        expectedVal1 = 'multiple'
        expectedVal2 = 'labels'

        p = Pool()
        p.add('foo.bar', expectedVal1)
        p.add('bar.foo', expectedVal2)

        self.assertEqualVector(p['foo.bar'], [expectedVal1])
        self.assertEqualVector(p['bar.foo'], [expectedVal2])

    def testStringVectorPoolMultiple(self):
        expectedVec1 = ['1.6', '0.9', '19.85']
        expectedVec2 = ['-5.0', '0.0', '5.0']

        p = Pool()
        p.add('foo.bar', expectedVec1)
        p.add('bar.foo', expectedVec2)

        self.assertEqualMatrix(p['foo.bar'], [expectedVec1])
        self.assertEqualMatrix(p['bar.foo'], [expectedVec2])

    def testDescriptorNames(self):
        key1 = 'foo.bar'
        key2 = 'bar.foo'
        key3 = 'foo.str'
        key4 = 'realvec'
        key5 = 'strvec'
        expected = [key1, key2, key3, key4, key5]
        expected.sort()

        p = Pool()
        p.add(key1, 20.08)
        p.add(key2, 20.09)
        p.add(key3, 'so many tests...')
        p.add(key4, [15, 567, 784, 0])
        p.add(key5, ['hello', 'world'])

        result = p.descriptorNames()
        result.sort()

        self.assertEqualVector(result, expected)

    def testDescriptorNames2(self):
        key1 = 'foo.bar'
        key2 = 'bar.foo'
        key3 = 'foo.str'
        key4 = 'realvec'
        key5 = 'strvec'
        expected = [key1, key3]
        expected.sort()

        p = Pool()
        p.add(key1, 20.08)
        p.add(key2, 20.09)
        p.add(key3, 'so many tests...')
        p.add(key4, [15, 567, 784, 0])
        p.add(key5, ['hello', 'world'])

        result = p.descriptorNames('foo')
        result.sort()

        self.assertEqualVector(result, expected)

    def testAddToDescriptorWithChildren(self):
        p = Pool()
        p.add('foo.bar', 54)

        self.assertRaises(RuntimeError, p.add, 'foo', 3.4)

    def testAddToDescriptorWithParentThatHasValues(self):
        # sorry for the long test name

        p = Pool()
        p.add('foo', 54)

        self.assertRaises(RuntimeError, p.add, 'foo.bar', 3.1459)

    def testAddMixedTypes(self):
        p = Pool()
        p.add('foo', 54)

        self.assertRaises(KeyError, p.add, 'foo', 'crap')

    def testBadIdea(self):
        p = Pool()
        p.add('foo.bar.cat.ugly', 32.3)

        self.assertRaises(RuntimeError, p.add, 'foo.bar.cat', 'its a string')


    # array2D tests:
    def testRealArray2DPoolSimple(self):
        expectedMat = array([[1,2],[3,4]])

        p = Pool()
        p.add('foo.bar', expectedMat)

        self.assertEqualMatrix(p['foo.bar'], [expectedMat])

    def testRealMatrixPoolMultiple(self):
        expectedMat1 = array([[1,2],[3,4]])
        expectedMat2 = array([[5,6],[7,8]])

        p = Pool()
        p.add('foo.bar', expectedMat1)
        p.add('foo.bar', expectedMat2)

        self.assertEqualMatrix(p['foo.bar'], [expectedMat1, expectedMat2])

    def testRealMatrixPoolMultipleLabels(self):
        rows = 2
        cols = 2
        expectedMat1 = zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                expectedMat1[i][j] = i*rows + j
        expectedMat2 = expectedMat1*5

        p = Pool()
        p.add('foo.bar', expectedMat1)
        p.add('bar.foo', expectedMat2)

        self.assertEqualMatrix(p['foo.bar'], [expectedMat1])
        self.assertEqualMatrix(p['bar.foo'], [expectedMat2])

    def testRealMatrixSeveralFrames(self):
        expected = [array([[0,1],[2,3]]),
                    array([[1,2],[3,4]]),
                    array([[2,3],[4,5]])]

        p = Pool()
        p.add('foo.bar', array([[0,1],[2,3]]))
        p.add('foo.bar', array([[1,2],[3,4]]))
        p.add('foo.bar', array([[2,3],[4,5]]))

        self.assertEqualMatrix(p['foo.bar'], expected)

    # Test adding an empty matrix
    def testVectorEmpty(self):
        p = Pool()
        p.add('foo.bar', zeros((10, 10)))
        p.add('foo.bar', array([[]]))
        self.assertEqualMatrix(p['foo.bar'], [zeros((10, 10)), array([[]])])

    def testStereoSample(self):
        p = Pool()
        p.add('foo', (3,6))

        self.assertEqual(len(p['foo']), 1)
        self.assertEqual(p['foo'][0][0], 3)
        self.assertEqual(p['foo'][0][1], 6)

        p.add('foo', (4,5))

        self.assertEqual(len(p['foo']), 2)
        self.assertEqual(p['foo'][1][0], 4)
        self.assertEqual(p['foo'][1][1], 5)
    def testValidityCheck(self):
        from numpy import nan as NaN
        from numpy import inf as infinity
        p =Pool()
        self.assertRaises(EssentiaException, p.add, 'foo', NaN, True)
        self.assertRaises(EssentiaException, p.add, 'foo', infinity, True)
        p.add('foo', 1, True)
        p.add('foo', NaN, False)
        p.add('foo', infinity, False)
        self.assertEqualVector(p.descriptorNames(), ['foo'])
        self.assertEqual(len(p['foo']), 3)
        # can't check for equality on Nan or inf

    def testMergeDescriptorSingleInt(self):
        p1 = Pool()
        p1.set('int', 1)
        p2 =Pool()
        p2.set('int', 2)
        # single value descriptors only allow for replace when merging data:
        self.assertRaises(EssentiaException, p1.mergeSingle,'int', p2['int'])
        p1.mergeSingle('int', p2['int'], 'replace')
        self.assertEqual( p1['int'], p2['int'])

    def testMergeDescriptorSingleReal(self):
        p1 = Pool()
        p1.set('real', 1)
        p2 =Pool()
        p2.set('real', 2)
        # single value descriptors only allow for replace when merging data:
        self.assertRaises(EssentiaException, p1.mergeSingle,'real', p2['real'])
        p1.mergeSingle('real', p2['real'], 'replace')
        self.assertEqual( p1['real'], p2['real'])

    def testMergeDescriptorSingleString(self):
        p1 = Pool()
        p1.set('string', '1')
        p2 =Pool()
        p2.set('string', '2')
        # single value descriptors only allow for replace when merging data:
        self.assertRaises(EssentiaException, p1.mergeSingle,'string', p2['string'])
        p1.mergeSingle('string', p2['string'], 'replace')
        self.assertEqual( p1['string'], p2['string'])

    def mergeDescriptor(self, p1, p2, name, dim):

        def copyPool(origin, to):
            for desc in origin.descriptorNames():
                if origin.isSingleValue(desc): to.set(desc, origin[desc])
                else:
                    for val in origin[desc]:
                        to.add(desc,val)

        def interleave(x,y):
            result = []
            for val in zip(x,y):
               for elem in val:
                   result.append(elem)
            return result

        p = Pool()
        copyPool(p1, p)

        if name in numpy.intersect1d(p1.descriptorNames(), p2.descriptorNames()):
            self.assertRaises(EssentiaException, p1.merge, name, p2[name])
        p1.merge(p2, 'replace')
        if not dim or dim == 0 and name in p1.descriptorNames() : # int, float, string
            self.assertEqual(p1[name], p2[name])
        elif dim==1: # vector_*
            self.assertEqualVector(p1[name], p2[name])
        elif dim==2: # vector_vector_*
            self.assertEqualMatrix(p1[name], p2[name])
        elif dim==3: # array2d
            for i in range(len(p1[name])):
                self.assertEqualMatrix(p1[name][i], p2[name][i])
        p1.clear()
        copyPool(p,p1)
        p1.merge(p2, 'append')
        if not dim or dim == 0  and name in p1.descriptorNames():# int, float, string
            self.assertRaises(EssentiaException, p1.merge, name, p2[name])
        elif dim==1:# vector_*
            self.assertEqualVector(p1[name], numpy.concatenate((p[name],p2[name])))
        elif dim==2:# vector_vector_*
            self.assertEqualMatrix(p1[name], numpy.concatenate((p[name],p2[name])))
        elif dim==3:# array2d
            for i in range(len(p1[name])):
                if i < len(p[name]): expected = p[name][i]
                else : expected = p2[name][i-len(p[name])]
                self.assertEqualMatrix(p1[name][i], expected);

        p1.clear()
        copyPool(p,p1)
        p1.merge(p2, 'interleave')
        if not dim or dim == 0 and name in p1.descriptorNames():# int, float, string
            self.assertRaises(EssentiaException, p1.merge, name, p2[name])
        elif dim==1:# vector_*
            self.assertEqualVector(p1[name], interleave(p[name],p2[name]))
        elif dim==2:# vector_vector_*
            self.assertEqualMatrix(p1[name], interleave(p[name],p2[name]))
        elif dim==3:# array2d
            expected = interleave(p[name], p2[name])
            for i in range(len(p1[name])):
                self.assertEqualMatrix(p1[name][i], expected[i])

    def testMergeDescriptorStringRealInt(self):
        def addToPool(name, offset, r, fun):
            p = Pool()
            for i in range(r):
                p.add(name, fun(offset+i))
            return p

        p1 = addToPool('int', 0, 3, int)
        p2 = addToPool('int', 10, 3, int)
        self.mergeDescriptor(p1, p2, 'int', 1)

        p1 = addToPool('real', 0, 3, float)
        p2 = addToPool('real', 10, 3, float)
        self.mergeDescriptor(p1, p2, 'real', 1)

        p1 = addToPool('string', 0, 3, str)
        p2 = addToPool('string', 10, 3, str)
        self.mergeDescriptor(p1, p2, 'string', 1)


    def testMergeVectorStringRealInt(self):
        def addToPool(name, offset, r, fun):
            p = Pool()
            vec = [[fun(offset+i*r), fun(offset+i*r+1), fun(offset+i*r+2)] for i in xrange(r)]
            for x in vec: p.add(name, x)
            return p

        p1 = addToPool('vector_int', 0, 3, int)
        p2 = addToPool('vector_int', 10, 3, int)
        self.mergeDescriptor(p1, p2, 'vector_int', 2)

        p1 = addToPool('vector_real', 0, 3, float)
        p2 = addToPool('vector_real', 10, 3, float)
        self.mergeDescriptor(p1, p2, 'vector_real', 2)

        p1 = addToPool('vector_string', 0, 3, str)
        p2 = addToPool('vector_string', 10, 3, str)
        self.mergeDescriptor(p1, p2, 'vector_string', 2)

    def testMergeArray2D(self):
        p1 = Pool()
        p1.add('array2d_real', numpy.array(numpy.arange(0,4).reshape([2,2]),dtype='float32'))
        p1.add('array2d_real', numpy.array(numpy.arange(4,8).reshape([2,2]), dtype='float32'))

        p2 = Pool()
        p2.add('array2d_real', numpy.array(numpy.arange(8,12).reshape([2,2]),dtype='float32'))
        p2.add('array2d_real', numpy.array(numpy.arange(12,16).reshape([2,2]), dtype='float32'))

        self.mergeDescriptor(p1, p2, 'array2d_real', 3)


    def testMergePool(self):
        def copyPool(origin, to):
            for desc in origin.descriptorNames():
                if origin.isSingleValue(desc):
                    to.set(desc, origin[desc])
                else:
                    for val in origin[desc]:
                        to.add(desc,val)


        p = Pool()
        p.set('single_int', 0)
        p.set('single_real', 0)
        p.set('single_string', '0')
        p.set('single_vector_real', [0.0, 1.0, 2.0])
        for i in range(3):
            p.add('int', i)
            p.add('real', float(i))
            p.add('string', str(i))

        for i in range(3):
            p.add('vector_int', [i*3+j for j in range(3)])
            p.add('vector_real', [float(i*3+j) for j in range(3)])
            p.add('vector_string', [str(i*3+j) for j in range(3)])

        p.add('array2d_real', numpy.array(numpy.arange(0,4).reshape([2,2]),dtype='float32'))
        p.add('array2d_real', numpy.array(numpy.arange(4,8).reshape([2,2]), dtype='float32'))

        p2 = Pool()
        p2.set('single_int', 3)
        p2.set('single_real', 3)
        p2.set('single_string', '3')
        p2.set('single_vector_real', [3.0, 4.0, 5.0])
        offset = 3
        for i in range(3):
            p2.add('int', offset+i)
            p2.add('real', float(offset+i))
            p2.add('string', str(offset+i))

        offset = 9
        for i in range(3):
            p2.add('vector_int', [offset+i*3+j for j in range(3)])
            p2.add('vector_real', [float(offset+i*3+j) for j in range(3)])
            p2.add('vector_string', [str(offset+i*3+j) for j in range(3)])

        p2.add('array2d_real', numpy.array(numpy.arange(8,12).reshape([2,2]),dtype='float32'))
        p2.add('array2d_real', numpy.array(numpy.arange(12,16).reshape([2,2]), dtype='float32'))

        p1 = Pool()
        copyPool(p, p1)
        # should fail when giving no merging type:
        self.assertRaises(EssentiaException, p1.merge, p2)
        # should fail in both append and interleave cause we have single value
        # descriptors:
        self.assertRaises(EssentiaException, p1.merge, p2, 'append')
        self.assertRaises(EssentiaException, p1.merge, p2, 'append')
        p1.merge(p2, 'replace')
        for desc in p1.descriptorNames():
            if 'single' in desc:
                if 'vector' not in desc: self.assertEqual(p1[desc], p2[desc])
                else: self.assertEqualVector(p1[desc], p2[desc])
            else:
                if 'vector' not in desc and 'array2d' not in desc:
                    self.assertEqualVector(p1[desc], p2[desc])
                if 'vector' in desc: self.assertEqualMatrix(p1[desc], p2[desc])
                if 'array2d' in desc:
                    for i in range(len(p1[desc])):
                        self.assertEqualMatrix(p1[desc][i], p2[desc][i])

        # remove single value descriptors from p so we can test append and
        # interleave:
        for desc in p.descriptorNames():
            if 'single' in desc: p.remove(desc)
        p1.clear()
        copyPool(p, p1)
        p1.merge(p2, 'append')
        for desc in p1.descriptorNames():
            # as we removed all single value descriptors from p, after appending all p2's
            # single values should have been added to p1
            if 'single' in desc:
                if 'vector' not in desc: self.assertEqual(p1[desc], p2[desc])
                else: self.assertEqualVector(p1[desc], p2[desc])
            else:
                expected = numpy.concatenate((p[desc],p2[desc]))
                if 'vector' not in desc and 'array2d' not in desc:
                    self.assertEqualVector(p1[desc], expected)
                if 'vector' in desc:
                    self.assertEqualMatrix(p1[desc], expected);
                if 'array2d' in desc:
                    for i in range(len(p1[desc])):
                        if i < len(p[desc]): expected = p[desc][i]
                        else : expected = p2[desc][i-len(p[desc])]
                        self.assertEqualMatrix(p1[desc][i], expected);

        # remove single value descriptors from p so we can test append and
        # interleave:
        def interleave(x,y):
            result = []
            for val in zip(x,y):
               for elem in val:
                   result.append(elem)
            return result

        for desc in p.descriptorNames():
            if 'single' in desc: p.remove(desc)
        p1.clear()
        copyPool(p, p1)
        p1.merge(p2, 'interleave')
        for desc in p1.descriptorNames():
            # as we removed all single value descriptors from p, after appending all p2's
            # single values should have been added to p1
            if 'single' in desc:
                if 'vector' not in desc: self.assertEqual(p1[desc], p2[desc])
                else: self.assertEqualVector(p1[desc], p2[desc])
            else:
                expected = interleave(p[desc],p2[desc])
                if 'vector' not in desc and 'array2d' not in desc:
                    self.assertEqualVector(p1[desc], expected)
                if 'vector' in desc:
                    self.assertEqualMatrix(p1[desc], expected);
                if 'array2d' in desc:
                    for i in range(len(p1[desc])):
                        self.assertEqualMatrix(p1[desc][i], expected[i]);




suite = allTests(TestPool)

if __name__ == '__main__':
    TextTestRunner(verbosity=2).run(suite)
