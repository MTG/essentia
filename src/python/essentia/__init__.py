import _essentia
import sys as _sys
from _essentia import reset
from common import Pool, array, ones, zeros
from translate import translate
import standard
from utils import *

#from progress import Progress


# version
__version__ = _essentia.version()


# base Exception class
class EssentiaError(Exception):
    def __init__(self, error, filename=None):
        self.error = error
        self.filename = filename
    def __str__(self):
        if self.filename:
            return "ERROR: " + self.error.replace(';',':') + self.filename
        else:
            return "ERROR: " + self.error.replace(';',':')


# set up logging system
# TODO: deprecate!
import logging as _logging

_logging.basicConfig(level = _logging.INFO,
                    format = '%(message)s')

INFO = _logging.info
logger = _logging.getLogger()


# debug level messages (separate handling because they are split into modules)
class DebuggingModule:
    EAlgorithm   = 1 << 0
    EConnectors  = 1 << 1
    EFactory     = 1 << 2
    ENetwork     = 1 << 3
    EGraph       = 1 << 4
    EExecution   = 1 << 5
    EMemory      = 1 << 6

    EPython      = 1 << 20
    EUnittest    = 1 << 21

    ENone        = 0
    EAll         = (1 << 30) - 1

# import these enums directly in our namespace for convenience
for dbg in dir(DebuggingModule):
    if not dbg.startswith('E'): continue
    setattr(_sys.modules[__name__], dbg, getattr(DebuggingModule, dbg))


class BitMask(object):
    def __init__(self, value = 0):
        self._value = value

    def __or__(self, other):
        self._value |= other

    def __add__(self, other):
        print 'add'
        return self._value | other

    def __iadd__(self, other):
        print 'iadd'
        self._value |= other
        return self

    def __and__(self, other):
        self._value &= other

    def __invert__(self):
        self._value = ~self._value

    def __sub__(self, other):
        return self._value & ~other

    def __isub__(self, other):
        self._value &= ~other
        return self

    def __int__(self):
        return self._value



def _getDebugLevel(_):
    return BitMask(_essentia.debugLevel())

def _setDebugLevel(_, levels):
    _essentia.setDebugLevel(int(levels))


# normal logging level messages
class EssentiaLogger(object):
    debug = property(_getDebugLevel, _setDebugLevel,
                     doc = 'A bit mask indicating which debugging modules should be activated')

    info = property(lambda l: _essentia.infoLevel(), lambda l, active: _essentia.setInfoLevel(active),
                    doc = 'A boolean indicating whether messages at the info level should be displayed')

    warning = property(lambda l: _essentia.warningLevel(), lambda l, active: _essentia.setWarningLevel(active),
                    doc = 'A boolean indicating whether messages at the warning level should be displayed')

    error = property(lambda l: _essentia.errorLevel(), lambda l, active: _essentia.setErrorLevel(active),
                    doc = 'A boolean indicating whether messages at the error level should be displayed')

log = EssentiaLogger()



from essentia.streaming import VectorInput

# we wrap this here so that we can do the decorator trick in all_tests.py
# FIXME: what decorator trick? is this comment still valid?
def run(gen):
    # catch this here as the actual type has not been determined yet so trying
    # run it here and now would result in an invalid pointer dereference...
    if isinstance(gen, VectorInput) and not gen.connections.values()[0]:
        raise EssentiaError('VectorInput is not connected to anything...')
    return _essentia.run(gen)
