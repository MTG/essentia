import _essentia
import sys as _sys
from _essentia import reset
from common import Pool, array, ones, zeros
from progress import Progress
from utils import *


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

# debug level messages (separate handling because they are split into modules)
class DebuggingModule:
    EAlgorithm   = 1 << 0
    EConnectors  = 1 << 1
    EFactory     = 1 << 2
    ENetwork     = 1 << 3
    EGraph       = 1 << 4
    EExecution   = 1 << 5
    EMemory      = 1 << 6
    EScheduler   = 1 << 7

    EPython      = 1 << 20
    EPyBindings  = 1 << 21
    EUnittest    = 1 << 22

    EUser1       = 1 << 25
    EUser2       = 1 << 26

    ENone        = 0
    EAll         = (1 << 30) - 1

# import the DebuggingModule enum directly in our namespace for convenience
for dbg in dir(DebuggingModule):
    if not dbg.startswith('E'): continue
    setattr(_sys.modules[__name__], dbg, getattr(DebuggingModule, dbg))


class BitMask(object):
    def __init__(self, value = 0):
        self._value = value

    def __or__(self, other):
        self._value |= other

    def __add__(self, other):
        return self._value | other

    def __iadd__(self, other):
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

    def __repr__(self):
        active, inactive = [], []
        for dbg in dir(DebuggingModule):
            if (not dbg.startswith('E') or
                dbg == 'EAll' or dbg == 'ENone'):
                continue
            if self._value & getattr(DebuggingModule, dbg):
                active.append(dbg[1:])
            else:
                inactive.append(dbg[1:])
        return 'DebugLevels(On=[%s], Off=[%s])' % (','.join(active), ','.join(inactive))



def _getDebugLevel(_):
    return BitMask(_essentia.debugLevel())

def _setDebugLevel(_, levels):
    _essentia.setDebugLevel(int(levels))


class EssentiaLogger(object):
    debugLevels = property(_getDebugLevel, _setDebugLevel,
                           doc = 'A bit mask indicating which debugging modules should be activated')

    infoActive = property(lambda l: _essentia.infoLevel(), lambda l, active: _essentia.setInfoLevel(active),
                          doc = 'A boolean indicating whether messages at the info level should be displayed')

    warningActive = property(lambda l: _essentia.warningLevel(), lambda l, active: _essentia.setWarningLevel(active),
                             doc = 'A boolean indicating whether messages at the warning level should be displayed')

    errorActive = property(lambda l: _essentia.errorLevel(), lambda l, active: _essentia.setErrorLevel(active),
                           doc = 'A boolean indicating whether messages at the error level should be displayed')

    @staticmethod
    def debug(level, s):
        _essentia.log_debug(level, s)

    @staticmethod
    def info(s):
        _essentia.log_info(s)

    @staticmethod
    def warning(s):
        _essentia.log_warning(s)

    @staticmethod
    def error(s):
        _essentia.log_error(s)

# Main logger to be used by the submodules
log = EssentiaLogger()

# FIXME: remove the use of INFO in favor of essentia.log.info
INFO = log.info

# we wrap this here so that we can do the decorator trick in all_tests.py
# FIXME: what decorator trick? is this comment still valid?
def run(gen):
    from essentia.streaming import VectorInput
    # catch this here as the actual type has not been determined yet so trying
    # run it here and now would result in an invalid pointer dereference...
    if isinstance(gen, VectorInput) and not gen.connections.values()[0]:
        raise EssentiaError('VectorInput is not connected to anything...')
    return _essentia.run(gen)

log.debug(EPython, 'Successfully imported essentia python module (log fully available and synchronized with the C++ one)')
