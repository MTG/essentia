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

from six import iteritems
from . import _essentia
import essentia
import sys as _sys
from . import common as _c
from ._essentia import skeys as algorithmNames, sinfo as algorithmInfo

# Used as a place-holder for sources and sinks, implements the right shift
# operator
class _StreamConnector:

    def __init__(self, input_algo, output_algo, name):
        '''
        input_algo - is the algo to be used if StreamConnector is interpreted as a sink (right)
        output_algo - is the algo to be used if StreamConnector is interpreted as a source (left)
        name - is the name of the sink/source

        input_algo or output_algo should be set to None if StreamConnector cannot be interpreted as
        an input or output, respectively.
        '''
        self.input_algo = input_algo
        self.output_algo = output_algo
        self.name = name

    def __rshift__(left, right):
        if not isinstance(left.output_algo, VectorInput) and \
           not left.output_algo.hasOutput(left.name):
            raise NameError('The \'%s\' algorithm does not have a source called \'%s\''
                            %(left.output_algo.name(), left.name))

        # connect a source to a sink
        if isinstance(right, _StreamConnector):
            # need to make an exception if the left algorithm is a VectorInput
            # because its underlying c++ type is initialized at connection time
            if isinstance(left.output_algo, VectorInput):
                left.output_algo.__inner_init__(_c.Edt(right.input_algo.getInputType(right.name)))

            if not right.input_algo.hasInput(right.name):
                raise NameError('The \'%s\' algorithm does not have a sink called \'%s\''
                                %(right.input_algo.name(), right.name))

            _essentia.connect(left.output_algo, left.name,
                              right.input_algo, right.name)

            # update connections
            left.output_algo.connections[left].append(right)

            return right


        elif isinstance(right, _essentia.StreamingAlgorithm) and right.name() == 'FileOutput':
            _essentia.fileOutputConnect(left.output_algo, left.name, right)

            # update connections
            left.output_algo.connections[left].append(right)

            return right


        # connect a source to a pool
        elif isinstance(right, tuple):
            if not len(right) == 2 or \
               not isinstance(right[0], _c.Pool) or \
               not isinstance(right[1], str):
                raise TypeError('the right side tuple given should consist of a Pool and a string descriptor name')

            # make sure to lazy-initialize the VectorInput
            if isinstance(left.output_algo, VectorInput):
                left.output_algo.__inner_init_default__()

            # update connections
            left.output_algo.connections[left].append(right)

            return _essentia.poolConnect(left.output_algo, left.name, right[0].cppPool, right[1])

        # connect a source to NOWHERE
        elif right is None:
            # still need to lazy-initialize the VectorInput
            if isinstance(left.output_algo, VectorInput):
                left.output_algo.__inner_init_default__()

            # update connections
            left.output_algo.connections[left].append(right)

            return _essentia.nowhereConnect(left.output_algo, left.name)

        # none of the above accepted types: raise an exception
        raise TypeError('\'%s.%s\' A source can only be connected to a sink, a pair '\
                        '(tuple) of pool and key name, or None'
                        %(left.output_algo.name(), left.name))

    def disconnect(self, connector):
        # if they are not connected just return
        if connector not in self.output_algo.connections[self]: return;
        # remove connector from connections
        self.output_algo.connections[self].remove(connector)

        # call internal disconnect based on connector type
        if isinstance(connector, _StreamConnector):
            return _essentia.disconnect(self.output_algo, self.name, connector.input_algo, connector.name)

        elif isinstance(connector, tuple):
            if not len(connector) == 2 or \
               not isinstance(connector[0], _c.Pool) or \
               not isinstance(connector[1], str):
                raise TypeError('the tuple given should consist of a Pool and a string descriptor name')

            return _essentia.poolDisconnect(self.output_algo, self.name, connector[0].cppPool, connector[1])

        elif connector == None:
            return _essentia.nowhereDisconnect(self.output_algo, self.name)

        raise TypeError('\'%s.%s\' An output can only be disconnected from an '\
                        'input, a pair (tuple) of pool and key name, or None'
                        %(self.output_algo.name(), self.name))

    def totalProduced(self):
        return _essentia.totalProduced(self.output_algo, self.name)



def _create_streaming_algo(givenname):
    essentia.log.debug(essentia.EPython, 'Creating essentia.streaming class: %s' % givenname)

    _algoInstance = _essentia.StreamingAlgorithm(givenname)
    _algoDoc = _algoInstance.getDoc()
    _algoStruct = _algoInstance.getStruct()
    del _algoInstance

    class StreamingAlgo(_essentia.StreamingAlgorithm):
        __doc__ = _algoDoc
        __struct__ = _algoStruct

        def __init__(self, **kwargs):
            _essentia.StreamingAlgorithm.__init__(self, givenname)

            self.configure(**kwargs)

            # keys should be StreamConnectors (outputs) and values should be lists
            # of StreamConnectors/pool tuples/None (inputs)
            self.connections = {}

            # populate instance members from sources and sinks
            # we don't descriminate b/w inputs and outputs at this point, put
            # into a set (i.e. no duplicates)
            names = set(self.inputNames())
            for outputName in self.outputNames():
                names.add(outputName)

            for n in names:
                # TODO self should only be used in the first arg if this StreamConnector should be
                # an input, and in the second arg position if this StreamConnector should be an
                # output. Setting both to self won't hurt though, afaik.
                conn = _StreamConnector(self, self, n)
                self.connections[conn] = []
                setattr(self, n, conn)

        def configure(self, **kwargs):
            # verify that all types match and do any necessary conversions
            for name, val in iteritems(kwargs):
                goalType = self.paramType(name)
                try:
                    convertedVal = _c.convertData(val, goalType)
                except TypeError as e:
                    raise TypeError('Error verifying \''+name+'\' parameter: '+str(e))

                kwargs[name] = convertedVal

            # we need to keep a reference to these parameters so that they won't get cleaned up for
            # the lifetime of this algo
            self.__pyparams__ = kwargs

            self.__configure__(**kwargs)

    algoClass = _c.algoDecorator(StreamingAlgo)
    setattr(_sys.modules[__name__], givenname, algoClass)


# load all streaming algorithms into module
def _reloadStreamingAlgorithms():
    for name in algorithmNames():
        _create_streaming_algo(name)

_reloadStreamingAlgorithms()

# This subclass provides some more functionality for VectorInput
class VectorInput(_essentia.VectorInput):
    __doc__ = 'VectorInput v1.0\n\n\n'+\
              '  Outputs:\n\n'+\
              '    [variable] data - the given data\n\n\n'+\
              '  Description:\n\n'+\
              '    Can be used as the starting point of a streaming network. Its constructor\n'+\
              '    takes in a vector that is to be streamed one token at a time.'
    __struct__ = {'category': 'Input/output',
                  'description': 'Can be used as the starting point of a streaming network. Its constructor\ntakes in a vector that is to be streamed one token at a time.',
                  'inputs': [],
                  'name': 'VectorInput',
                  'outputs': [{'description': 'the given data',
                               'name': 'data',
                               'type': 'vector_<real>'}],
                  'parameters': []}


    def __init__(self, data):
        self.__initialized = False
        self.__initializedType = _c.Edt(_c.Edt.UNDEFINED)
        self.dataref = data
        self.data = _StreamConnector(None, self, 'data')

        # keys should be StreamConnectors (outputs) and values should be lists
        # of StreamConnectors/pool tuples/None (inputs)
        self.connections = {}

        self.connections[self.data] = []

    # initializes the underlying c++ type
    # sinkEdt represents the type that the VectorInput should be initialized to
    # (e.g. the type of the sink we're connecting this to)
    def __inner_init__(self, sinkEdt):
        # get vector form of sinkEdt, because we'll compare to given data input
        # which is a vector of something
        sinkEdt = sinkEdt.vectorize()

        if self.__initialized:
            if self.__initializedType == sinkEdt:
                return
            raise TypeError('VectorInput was already connected to another sink with a different type, original type: '+str(self.__initializedType)+' new type: '+str(sinkEdt))

        self.dataref = _c.convertData(self.dataref, sinkEdt)
        _essentia.VectorInput.__init__(self, self.dataref, str(sinkEdt))
        self.__initializedType = sinkEdt #_c.Edt
        self.__initialized = True


    # This method should be used if the sink being connected to is also
    # ambiguous (e.g. Pool or NOWHERE). This method will guess the type of the
    # VectorInput (a type that would be valid for a pool (e.g. VECTOR_REAL,
    # VECTOR_STRING)) and raise a TypeError if a type could not be guessed.
    def __inner_init_default__(self):
        if self.__initialized:
            return

        sourceEdt = _c.determineEdt(self.dataref)

        if not sourceEdt.isIntermediate():
            _essentia.VectorInput.__init__(self, self.dataref, str(sourceEdt))
            self.__initialized = True
            return

        if sourceEdt == _c.Edt.LIST_EMPTY or \
           sourceEdt == _c.Edt.LIST_INTEGER or \
           sourceEdt == _c.Edt.LIST_REAL or \
           sourceEdt == _c.Edt.LIST_MIXED:
            self.dataref = _c.convertData(self.dataref, _c.Edt.VECTOR_REAL)
            _essentia.VectorInput.__init__(self, self.dataref, _c.Edt.VECTOR_REAL)
            self.__initialized = True
            return

        if sourceEdt == _c.Edt.LIST_LIST_REAL or sourceEdt == _c.Edt.LIST_LIST_INTEGER:
            self.dataref = _c.convertData(self.dataref, _c.Edt.MATRIX_REAL)
            _essentia.VectorInput.__init__(self, self.dataref, _c.Edt.MATRIX_REAL)
            self.__initialized = True
            return

        raise TypeError('Unable to initialize VectorInput because it is '+\
                        'being connected to None or a Pool, and the '+\
                        'VectorInput\'s data consists of an unsupported Pool '+\
                        'type: '+str(sourceEdt))

class CompositeBase(object):
    '''
    Inherit from this class when creating a new composite streaming algorithm.
    '''

    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def name(self): return self.__class__.__name__

    def hasInput(self, name):
        return name in self.inputs

    def hasOutput(self, name):
        return name in self.outputs

    def __getattr__(self, name):
        '''
        returns a _StreamConnector that can be used for connecting to inputs/outputs
        '''
        # just return the streamconnector of the real algorithm which this
        # streamconnector is pointing to:
        if self.hasInput(name) and  self.hasOutput(name):
            raise NameError('CompositeBase algorithms cannot '\
                            'have a sink and a source with the same '\
                            'name:\''+self.name()+'.'+name+'\'')
        elif self.hasInput(name):  return self.inputs[name]
        elif self.hasOutput(name): return self.outputs[name]
        raise NameError('The '+self.name()+ ' CompositeBase algorithm does not have a connector named \''+name+'\'')
