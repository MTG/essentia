import common as _c
import _essentia

def isSilent(arg):
    return _essentia.isSilent( _c.convertData(arg, _c.Edt.VECTOR_REAL) )

def instantPower(arg):
    return _essentia.instantPower( _c.convertData(arg, _c.Edt.VECTOR_REAL) )

def nextPowerTwo(arg):
    return _essentia.nextPowerTwo(_c.convertData(arg, _c.Edt.REAL))

def isPowerTwo(arg):
    return _essentia.isPowerTwo(arg)

def lin2db(arg):
    return _essentia.lin2db( _c.convertData(arg, _c.Edt.REAL) )

def db2lin(arg):
    return _essentia.db2lin( _c.convertData(arg, _c.Edt.REAL) )

def pow2db(arg):
    return _essentia.pow2db( _c.convertData(arg, _c.Edt.REAL) )

def db2pow(arg):
    return _essentia.db2pow( _c.convertData(arg, _c.Edt.REAL) )

def amp2db(arg):
    return _essentia.amp2db( _c.convertData(arg, _c.Edt.REAL) )

def db2amp(arg):
    return _essentia.db2amp( _c.convertData(arg, _c.Edt.REAL) )

def bark2hz(arg):
    return _essentia.bark2hz( _c.convertData(arg, _c.Edt.REAL) )

def hz2bark(arg):
    return _essentia.hz2bark( _c.convertData(arg, _c.Edt.REAL) )

def mel2hz(arg):
    return _essentia.mel2hz( _c.convertData(arg, _c.Edt.REAL) )

def hz2mel(arg):
    return _essentia.hz2mel( _c.convertData(arg, _c.Edt.REAL) )

def postProcessTicks(arg1, arg2=None, arg3=None):
    if arg2 != None and arg3 != None:
        return _essentia.postProcessTicks(_c.convertData(arg1, _c.Edt.VECTOR_REAL),
                                          _c.convertData(arg2, _c.Edt.VECTOR_REAL),
                                          _c.convertData(arg3, _c.Edt.REAL))
    return _essentia.postProcessTicks(_c.convertData(arg1, _c.Edt.VECTOR_REAL))

def normalize(array):
    return _essentia.normalize(_c.convertData(array, _c.Edt.VECTOR_REAL))

def derivative(array):
    return _essentia.derivative(_c.convertData(array, _c.Edt.VECTOR_REAL))

__all__ = [ 'isSilent', 'instantPower',
            'nextPowerTwo', 'isPowerTwo',
            'lin2db', 'db2lin',
            'pow2db', 'db2pow',
            'amp2db', 'db2amp',
            'bark2hz', 'hz2bark',
            'mel2hz', 'hz2mel',
            'postProcessTicks',
            'normalize', 'derivative']
