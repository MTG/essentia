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

import _essentia
import essentia
import common as _c
import sys as _sys
from _essentia import keys as algorithmNames, info as algorithmInfo


# given an essentia algorithm name, create the corresponding class
def _create_essentia_class(name, moduleName = __name__):
    essentia.log.debug(essentia.EPython, 'Creating essentia.standard class: %s' % name)

    _algoInstance = _essentia.Algorithm(name)
    _algoDoc = _algoInstance.getDoc()
    _algoStruct = _algoInstance.getStruct()
    del _algoInstance

    class Algo(_essentia.Algorithm):
        __doc__ = _algoDoc
        __struct__ = _algoStruct

        def __init__(self, **kwargs):
            # init the internal cpp wrapper
            _essentia.Algorithm.__init__(self, name)

            # configure the algorithm
            self.configure(**kwargs)

        def configure(self, **kwargs):
            # verify that all types match and do any necessary conversions
            for name, val in kwargs.iteritems():
                goalType = self.paramType(name)
                try:
                    convertedVal = _c.convertData(val, goalType)
                except TypeError: # as e: # catching exception as sth is only
                                          #available as from python 2.6
                    raise TypeError('Error cannot convert parameter %s to %s'\
                                    %(str(_c.determineEdt(val)),str(goalType))) #\''+name+'\' parameter: '+str(e))

                kwargs[name] = convertedVal

            self.__configure__(**kwargs)

        def compute(self, *args):
            inputNames = self.inputNames()

            if len(args) != len(inputNames):
                raise ValueError(name+'.compute requires '+str(len(inputNames))+' argument(s), '+str(len(args))+' given')

            # we have to make some exceptions for YamlOutput and PoolAggregator
            # because they expect cpp Pools
            if name in ('YamlOutput', 'PoolAggregator', 'SvmClassifier', 'PCA', 'GaiaTransform'):
                args = (args[0].cppPool,)

            # verify that all types match and do any necessary conversions
            result = []

            convertedArgs = []
            for i in range(len(inputNames)):
                goalType = _c.Edt(self.inputType(inputNames[i]))
                try:
                    convertedData = _c.convertData(args[i], goalType)
                except TypeError:
                    raise TypeError('Error cannot convert argument %s to %s' \
                          %(str(_c.determineEdt(args[i])), str(goalType)))

                convertedArgs.append(convertedData)

            results = self.__compute__(*convertedArgs)

            # we have to make an exceptional case for YamlInput, because we need
            # to wrap the Pool that it outputs w/ our python Pool from common.py
            if name in ('YamlInput', 'PoolAggregator', 'SvmClassifier', 'PCA', 'GaiaTransform', 'Extractor'):
                return _c.Pool(results)

            else:
                return results

        def __call__(self, *args):
            return self.compute(*args)

        def __str__(self):
            return __doc__


    algoClass = _c.algoDecorator(Algo)

    setattr(_sys.modules[moduleName], name, algoClass)

    return algoClass


# load all classes into python
def _reloadAlgorithms(moduleName = __name__):
    for name in _essentia.keys():
        _create_essentia_class(name, moduleName)

_reloadAlgorithms()


# load derived descriptors and other ones written in python
from algorithms import create_python_algorithms as _create_python_algorithms
_create_python_algorithms(_sys.modules[__name__])
