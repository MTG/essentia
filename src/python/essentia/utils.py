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

from . import common as _c
from . import _essentia

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

def lin2log(arg):
    return _essentia.lin2log( _c.convertData(arg, _c.Edt.REAL) )

def bark2hz(arg):
    return _essentia.bark2hz( _c.convertData(arg, _c.Edt.REAL) )

def hz2bark(arg):
    return _essentia.hz2bark( _c.convertData(arg, _c.Edt.REAL) )

def mel2hz(arg):
    return _essentia.mel2hz( _c.convertData(arg, _c.Edt.REAL) )

def hz2mel(arg):
    return _essentia.hz2mel( _c.convertData(arg, _c.Edt.REAL) )

def midi2hz(arg1, arg2=440.0):
    return _essentia.midi2hz( _c.convertData(arg1, _c.Edt.INTEGER),
                              _c.convertData(arg2, _c.Edt.REAL) )

def hz2midi(arg1, arg2=440.0):
    return _essentia.hz2midi( _c.convertData(arg1, _c.Edt.REAL),
                              _c.convertData(arg2, _c.Edt.REAL) )

def cents2hz(arg1, arg2):
    return _essentia.cents2hz(_c.convertData(arg1, _c.Edt.REAL),
                              _c.convertData(arg2, _c.Edt.REAL) )

def hz2cents(arg1, arg2):
    return _essentia.hz2cents(_c.convertData(arg1, _c.Edt.REAL),
                              _c.convertData(arg2, _c.Edt.REAL) )

def midi2note(arg):
    return _essentia.midi2note( _c.convertData(arg, _c.Edt.INTEGER) )

def note2midi(arg):
    return _essentia.note2midi( _c.convertData(arg, _c.Edt.STRING) )

def note2root(arg):
    return _essentia.note2root( _c.convertData(arg, _c.Edt.STRING) )

def note2octave(arg):
    return _essentia.note2octave( _c.convertData(arg, _c.Edt.STRING) )

def hz2note(arg1, arg2=440.0):
    return _essentia.hz2note( _c.convertData(arg1, _c.Edt.REAL),
                              _c.convertData(arg2, _c.Edt.REAL) )

def note2hz(arg1, arg2=440.0):
    return _essentia.note2hz( _c.convertData(arg1, _c.Edt.STRING),
                              _c.convertData(arg2, _c.Edt.REAL) )

def velocity2db(arg1, arg2=-96):
    return _essentia.velocity2db( _c.convertData(arg1, _c.Edt.INTEGER),
                                  _c.convertData(arg2, _c.Edt.REAL) )

def db2velocity(arg1, arg2=-96):
    return _essentia.db2velocity( _c.convertData(arg1, _c.Edt.REAL),
                                  _c.convertData(arg2, _c.Edt.REAL) )

def equivalentKey(arg):
    return _essentia.equivalentKey( _c.convertData(arg, _c.Edt.STRING) )

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
            'midi2hz', 'hz2midi',
	    'cents2hz', 'hz2cents',
	    'note2root', 'note2octave',
            'midi2note', 'note2midi',
            'hz2note', 'note2hz',
            'velocity2db', 'db2velocity',
            'postProcessTicks',
            'normalize', 'derivative',
            'equivalentKey', 'lin2log']
