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

import numpy
from six import iteritems
from . import _essentia


# force the array objects to be of type float32
def array(object, **kwargs):
    return numpy.array(object, dtype='f4', **kwargs)


def zeros(object, **kwargs):
    return numpy.zeros(object, dtype='f4', **kwargs)


def ones(object, **kwargs):
    return numpy.ones(object, dtype='f4', **kwargs)


algoDecorator = lambda x: x


# An object representing an enum which contains int representations for
# essentia types. The purpose of this int representation is to have a common
# space for which to compare python and c++ types that are relevant to Essentia
class Edt:  # Essentia Data Type
    # c++ types
    BOOL = 'BOOL'
    INTEGER = 'INTEGER'
    REAL = 'REAL'
    STRING = 'STRING'
    STEREOSAMPLE = 'STEREOSAMPLE'
    COMPLEX = 'COMPLEX'
    VECTOR_INTEGER = 'VECTOR_INTEGER'
    VECTOR_REAL = 'VECTOR_REAL'
    VECTOR_STRING = 'VECTOR_STRING'
    VECTOR_COMPLEX = 'VECTOR_COMPLEX'
    VECTOR_VECTOR_STRING = 'VECTOR_VECTOR_STRING'
    VECTOR_VECTOR_REAL = 'VECTOR_VECTOR_REAL'
    VECTOR_VECTOR_COMPLEX = 'VECTOR_VECTOR_COMPLEX'
    VECTOR_STEREOSAMPLE = 'VECTOR_STEREOSAMPLE'
    MATRIX_REAL = 'MATRIX_REAL'
    MATRIX_COMPLEX = 'MATRIX_COMPLEX'
    VECTOR_MATRIX_REAL = 'VECTOR_MATRIX_REAL'
    VECTOR_TENSOR_REAL = 'VECTOR_TENSOR_REAL'
    TENSOR_REAL = 'TENSOR_REAL'
    POOL = 'POOL'

    # intermediate types
    LIST_EMPTY = 'LIST_EMPTY'
    LIST_MIXED = 'LIST_MIXED'
    LIST_INTEGER = 'LIST_INTEGER'
    LIST_REAL = 'LIST_REAL'
    LIST_LIST_REAL = 'LIST_LIST_REAL'
    LIST_LIST_INTEGER = 'LIST_LIST_INTEGER'
    LIST_LIST_EMPTY = 'LIST_LIST_EMPTY'
    LIST_COMPLEX = 'LIST_COMPLEX'
    LIST_LIST_COMPLEX = 'LIST_LIST_COMPLEX'
    LIST_ARRAY_REAL = 'LIST_ARRAY_REAL'
    LIST_ARRAY = 'LIST_ARRAY'
    NUMPY_FLOAT = 'NUMPY_FLOAT'
    UNDEFINED = 'UNDEFINED'

    def __init__(self, tp):
        self._tp = tp

    def isIntermediate(self):
        return self._tp in (Edt.LIST_EMPTY, Edt.LIST_MIXED, Edt.LIST_INTEGER,
                            Edt.LIST_REAL, Edt.LIST_LIST_REAL,
                            Edt.LIST_LIST_INTEGER, Edt.LIST_ARRAY,
                            Edt.LIST_ARRAY_REAL, Edt.UNDEFINED, Edt.NUMPY_FLOAT,
                            Edt.LIST_LIST_EMPTY, Edt.LIST_LIST_COMPLEX)

    def vectorize(self):
        return Edt('VECTOR_'+self._tp)

    def devectorize(self):
        if not self._tp.startswith('VECTOR_'):
            raise ValueError('cannot devectorize an Edt that is not a vector (Edt='+self._tp+')')

        return Edt(self._tp[len('VECTOR_'):])

    def __eq__(self, other):
        if isinstance(other, str):
            return self._tp == other
        elif isinstance(other, Edt):
            return self._tp == other._tp
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self._tp


# Determines the essentia data type of a given python object
def determineEdt(obj):
    # lists
    if isinstance(obj, list):
        if len(obj) == 0:
            return Edt(Edt.LIST_EMPTY)

        firstElmtType = determineEdt(obj[0])

        for item in obj[1:]:
            if determineEdt(item) != firstElmtType:
                return Edt(Edt.LIST_MIXED)

        if firstElmtType == Edt.INTEGER:
            return Edt(Edt.LIST_INTEGER)

        if firstElmtType == Edt.REAL:
            return Edt(Edt.LIST_REAL)

        if firstElmtType == Edt.STRING:
            return Edt(Edt.VECTOR_STRING)

        if firstElmtType == Edt.COMPLEX:
            return Edt(Edt.LIST_COMPLEX)

        if firstElmtType == Edt.VECTOR_STRING:
            return Edt(Edt.VECTOR_VECTOR_STRING)

        if firstElmtType == Edt.STEREOSAMPLE:
            raise TypeError('received a list of StereoSamples, if a VECTOR_STEREOSAMPLE is desired, create a 2-dimensional numpy.array of dimensions Nx2')

        if firstElmtType == Edt.MATRIX_REAL:
            return Edt(Edt.VECTOR_MATRIX_REAL)

        if firstElmtType == Edt.TENSOR_REAL:
            return Edt(Edt.VECTOR_TENSOR_REAL)

        if firstElmtType == Edt.LIST_REAL:
            return Edt(Edt.LIST_LIST_REAL)

        if firstElmtType == Edt.LIST_COMPLEX:
            return Edt(Edt.LIST_LIST_COMPLEX)

        if firstElmtType == Edt.LIST_INTEGER:
            return Edt(Edt.LIST_LIST_INTEGER)

        if firstElmtType == Edt.LIST_EMPTY:
            return Edt(Edt.LIST_LIST_EMPTY)

        if isinstance(obj[0], numpy.ndarray) and obj[0].ndim == 1:
            if obj[0].dtype == numpy.dtype('single'):
                return Edt(Edt.LIST_ARRAY_REAL)
            else:
                return Edt(Edt.LIST_ARRAY)

    if isinstance(obj, numpy.ndarray) and obj.ndim == 4:
        if obj.dtype == numpy.dtype('single'):
            return Edt(Edt.TENSOR_REAL)

        if obj.dtype == numpy.dtype('complex64'):
            return Edt(Edt.TENSOR_COMPLEX)

        raise TypeError('essentia can currently only accept two-dimensional numpy arrays of dtype '\
                        '"single"')

    # numpy array matrices
    if isinstance(obj, numpy.ndarray) and obj.ndim == 2:
        if obj.dtype == numpy.dtype('single'):
            return Edt(Edt.MATRIX_REAL)

        if obj.dtype == numpy.dtype('complex64'):
            return Edt(Edt.MATRIX_COMPLEX)

        raise TypeError('essentia can currently only accept two-dimensional numpy arrays of dtype '\
                        '"single"')

    # numpy arrays
    if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
        if obj.dtype == numpy.dtype('single'):
            return Edt(Edt.VECTOR_REAL)
        if obj.dtype == numpy.dtype('int'):
            return Edt(Edt.VECTOR_INTEGER)
        if obj.dtype == numpy.dtype('complex64'):
            return Edt(Edt.VECTOR_COMPLEX)

        raise TypeError('essentia can currently only accept one-dimensional numpy arrays of dtype '\
                        '{"single", "int", "complex64"}, please consider using essentia.array to '\
                        'create your arrays')

    # bool (must go before ints! because True and False can be ints)
    if isinstance(obj, bool):
        return Edt(Edt.BOOL)

    # ints
    if isinstance(obj, int):
        return Edt(Edt.INTEGER)

    # reals
    if isinstance(obj, float):
        return Edt(Edt.REAL)

    # strings
    if isinstance(obj, str):
        return Edt(Edt.STRING)

    if isinstance(obj, numpy.complex64):
        return Edt(Edt.COMPLEX)

    if isinstance(obj, complex):
        return Edt(Edt.COMPLEX)

    if isinstance(obj, numpy.float32):
        return Edt(Edt.NUMPY_FLOAT)

    if isinstance(obj, dict):
        # map parameters
        if len(obj) > 0:
            firstType = None
            allKeysAreStrings = True
            allTypesEqual = True
            for key, val in iteritems(obj):
                if not isinstance(key, str):
                    allKeysAreStrings = False
                    break

                if firstType is None:
                    firstType = determineEdt(val)

                elif firstType != determineEdt(val):
                    allTypesEqual = False
                    break

            if allKeysAreStrings and allTypesEqual:
                return Edt('MAP_'+str(firstType))

    # pools
    if isinstance(obj, Pool) or isinstance(obj, _essentia.Pool):
        return Edt(Edt.POOL)

    # tuples
    if isinstance(obj, tuple) and len(obj) == 2 and \
       (isinstance(obj[0], float) or isinstance(obj[0], int)) and \
       (isinstance(obj[1], float) or isinstance(obj[1], int)):
        return Edt(Edt.STEREOSAMPLE)

    # everything else
    return Edt(Edt.UNDEFINED)


# Converts 'data' to 'goalType'. 'goalType' must be a non-intermediate EDT. If
# a conversion cannot be made, a TypeError will be raised.
def convertData(data, goalType):
    origType = determineEdt(data)

    if origType == goalType:
        return data

    if goalType == Edt.VECTOR_REAL:
        if origType == Edt.LIST_REAL or origType == Edt.LIST_INTEGER:
            return array(data)

        if origType == Edt.LIST_MIXED:
            for item in data:
                itemType = determineEdt(item)
                if not (itemType == Edt.REAL or itemType == Edt.INTEGER):
                    raise TypeError('Cannot convert data from type LIST_MIXED to type VECTOR_REAL ' +
                                    'because LIST_MIXED contains items not of type REAL or INTEGER (e.g. ' +
                                    str(itemType)+')')
            return array(data)

    if goalType == Edt.VECTOR_INTEGER:
        if origType == Edt.LIST_INTEGER:
            return numpy.array(data, numpy.int32)

    if goalType == Edt.VECTOR_REAL:
        if origType == Edt.LIST_INTEGER or\
           origType == Edt.VECTOR_INTEGER:
            return numpy.array(data, numpy.float32)

    if origType == Edt.LIST_EMPTY:
        if goalType == Edt.VECTOR_REAL:
            return array(data)
        if goalType == Edt.VECTOR_STRING:
            return data
        if goalType == Edt.VECTOR_INTEGER:
            return numpy.array(data, numpy.dtype('int'))
        if goalType == Edt.VECTOR_STEREOSAMPLE:
            return data

    if goalType == Edt.MATRIX_REAL and \
       (origType == Edt.LIST_LIST_REAL or origType == Edt.LIST_LIST_INTEGER or origType == Edt.LIST_ARRAY_REAL):
        return array(data)

    if goalType == Edt.VECTOR_VECTOR_REAL:
        if origType == Edt.MATRIX_REAL or origType == Edt.LIST_LIST_INTEGER:
            return [[float(col) for col in row] for row in data]

        if origType == Edt.LIST_LIST_REAL or origType == Edt.LIST_LIST_EMPTY or Edt.LIST_ARRAY_REAL:
            return data

    if goalType == Edt.REAL:
        if origType == Edt.INTEGER or origType == Edt.NUMPY_FLOAT:
            return float(data)

    if goalType == Edt.VECTOR_STEREOSAMPLE:
        if origType == Edt.LIST_LIST_INTEGER:
            for row in data:
                if len(row) != 2:
                    ValueError('Cannot convert a LIST_LIST_INTEGER to a VECTOR_STEREOSAMPLE if the sub-lists are not of length 2')
            return array(data)

        if origType == Edt.MATRIX_REAL:
            if data.shape[1] != 2:
                raise TypeError('Cannot convert a MATRIX_REAL to a VECTOR_STEREOSAMPLE if the second dimension of the MATRIX_REAL is not 2')

            return data

        if origType == Edt.LIST_MIXED:
            for row in data:
                if len(row) != 2:
                    ValueError('Cannot convert a LIST_MIXED to a VECTOR_STEREOSAMPLE if the sub-lists are not of length 2')
                try:
                    convertData(row, Edt.VECTOR_REAL)
                except:
                    TypeError('Cannot convert a LIST_MIXED to a VECTOR_STEREOSAMPLE if the sub-lists are not convertible to VECTOR_REAL')

            return array(data)

    if goalType == Edt.VECTOR_VECTOR_COMPLEX:
        if origType  == Edt.LIST_LIST_COMPLEX:
            return data

        if origType  == Edt.MATRIX_COMPLEX:
            return [[col for col in row] for row in data]

    raise TypeError('Cannot convert data from type %s (%s) to type %s' %
                    (str(origType), str(type(data)), str(goalType)))


class Pool:
    def __init__(self, poolRep=None):
        if poolRep is None:
            self.cppPool = _essentia.Pool()

        elif isinstance(poolRep, _essentia.Pool):
            self.cppPool = poolRep

        elif isinstance(poolRep, dict):
            self.cppPool = _essentia.Pool()
            for key, val in iteritems(poolRep):
                for v in val:
                    self.add(key, v)

        else:
            raise TypeError('poolRep argument must be a Cpp Essentia Pool or python dictionary')

    def add(self, key, value, validityCheck=False):
        givenType = determineEdt(value)

        # if we've seen this key before, determine the type
        if self.containsKey(key):
            goalType = Edt(self.cppPool.__keyType__(key)).devectorize()

        # if we haven't seen this type before, we will have to guess its type
        else:
            if givenType in (Edt.REAL, Edt.STRING, Edt.STEREOSAMPLE,
                             Edt.VECTOR_REAL, Edt.VECTOR_STRING,
                             Edt.VECTOR_STEREOSAMPLE, Edt.MATRIX_REAL,
                             Edt.TENSOR_REAL):
                goalType = givenType

            # some exceptions
            elif givenType == Edt.INTEGER:
                goalType = Edt(Edt.REAL)
            elif givenType == Edt.NUMPY_FLOAT:
                goalType = Edt(Edt.REAL)
            elif givenType == Edt.LIST_REAL:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.LIST_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.VECTOR_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)

            else:
                raise TypeError('Pool.add does not support the type: '+str(givenType))

        # try to convert the type
        try:
            convertedVal = convertData(value, goalType)
        except TypeError:
            raise KeyError('Pool.add could not convert given data to the type already in the Pool under the key \''+key+'\'')

        self.cppPool.__add__(key, str(goalType), convertedVal, validityCheck)

    def set(self, key, value, validityCheck=False):
        givenType = determineEdt(value)

        # if we've seen this key before, determine the type
        if self.containsKey(key):
            goalType = self.cppPool.__keyType__(key)

        # if we haven't seen this type before, we will have to guess its type
        else:
            if givenType in (Edt.REAL, Edt.STRING, Edt.VECTOR_REAL, Edt.TENSOR_REAL):
                goalType = givenType

            # some exceptions
            elif givenType == Edt.INTEGER:
                goalType = Edt(Edt.REAL)
            elif givenType == Edt.NUMPY_FLOAT:
                goalType = Edt(Edt.REAL)
            elif givenType == Edt.VECTOR_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.LIST_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.LIST_REAL:
                goalType = Edt(Edt.VECTOR_REAL)

            else:
                raise TypeError('Pool.set does not support the type: '+str(givenType))

        # try to convert the type
        try:
            convertedVal = convertData(value, goalType)
        except TypeError:
            raise KeyError('Pool.set could not convert given data to the type already in the Pool under the key \''+key+'\'')

        return self.cppPool.__set__(key, str(goalType), convertedVal, validityCheck)

    def merge(self, arg1, arg2=None, arg3=None):
        # this function is used to merge both descriptors and pools
        if determineEdt(arg1) == Edt.POOL:
            if arg3:
                raise TypeError('Pool.merge requires only 3 arguments when ' +
                                'merging entire pools: self, pool, string')
            if not arg2:
                mergeType = ''
            else:
                mergeType = arg2
            return self.cppPool.__merge__(str(Edt(Edt.POOL)), arg1.cppPool, mergeType)

        key = arg1
        value = arg2
        mergeType = arg3
        if not mergeType:
            mergeType = ''

        givenType = determineEdt(value)

        # if we've seen this key before, determine the type
        if self.containsKey(key):
            goalType = Edt(self.cppPool.__keyType__(key))  #.devectorize()

        # if we haven't seen this type before, we will have to guess its type
        else:
            if givenType in (Edt.REAL, Edt.STRING, Edt.STEREOSAMPLE,
                             Edt.VECTOR_REAL, Edt.VECTOR_STRING,
                             Edt.VECTOR_STEREOSAMPLE, Edt.MATRIX_REAL,
                             Edt.VECTOR_VECTOR_REAL, Edt.VECTOR_MATRIX_REAL,
                             Edt.VECTOR_TENSOR_REAL, Edt.TENSOR_REAL):
                goalType = givenType

            # some exceptions
            elif givenType == Edt.INTEGER:
                goalType = Edt(Edt.REAL)
            elif givenType == Edt.LIST_REAL:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.LIST_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.VECTOR_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)

            else:
                raise TypeError('Pool.merge does not support the type: '+str(givenType))

        # try to convert the type
        try:
            convertedVal = convertData(value, goalType)
        except TypeError:
            raise KeyError('Pool.merge could not convert given data to the type already in the Pool under the key\''+key+'\'')

        return self.cppPool.__merge__(key, str(goalType), convertedVal, mergeType)

    def mergeSingle(self, key, value, mergeType=''):
        givenType = determineEdt(value)

        # if we've seen this key before, determine the type
        if self.containsKey(key):
            goalType = self.cppPool.__keyType__(key)

        # if we haven't seen this type before, we will have to guess its type
        else:
            if givenType in (Edt.REAL, Edt.STRING, Edt.VECTOR_REAL):
                goalType = givenType

            # some exceptions
            elif givenType == Edt.INTEGER:
                goalType = Edt(Edt.REAL)
            elif givenType == Edt.VECTOR_INTEGER:
                goalType = Edt(Edt.VECTOR_REAL)
            elif givenType == Edt.NUMPY_FLOAT:
                goalType = Edt(Edt.REAL)

            else:
                raise TypeError('Pool.mergeSingle does not support the type: '+str(givenType))

        # try to convert the type
        try:
            convertedVal = convertData(value, goalType)
        except TypeError:
            raise KeyError('Pool.mergeSingle could not convert given data to the type already in the Pool under the key \''+key+'\'')

        return self.cppPool.__mergeSingle__(key, str(goalType), convertedVal, mergeType)

    def __getitem__(self, key):
        if not self.containsKey(key):
            raise KeyError('no key found named \''+key+'\'')

        return self.cppPool.__value__(key, self.cppPool.__keyType__(key))

    def containsKey(self, key):
        return key in self.descriptorNames()

    def descriptorNames(self, key=None):
        if not key:
            return self.cppPool.descriptorNames()
        return self.cppPool.descriptorNames(key)

    def remove(self, key):
        return self.cppPool.remove(key)

    def removeNamespace(self, namespace):
        return self.cppPool.removeNamespace(namespace)

    def isSingleValue(self, name):
        return self.cppPool.isSingleValue(name)

    def clear(self):
        return self.cppPool.clear()
