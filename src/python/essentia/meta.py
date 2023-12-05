from json import dump
from os.path import join
import essentia
from ._essentia import Algorithm, StreamingAlgorithm, keys, skeys


def _metadata_standard():
    meta = {}
    for name in keys():
        essentia.log.debug(essentia.EPython, 'Loading __doc__ and __struct__ metadata for essentia.standard class: %s' % name)
        _algoInstance = Algorithm(name)
        meta[name] = {}
        meta[name]['__doc__'] = _algoInstance.getDoc()
        meta[name]['__struct__'] = _algoInstance.getStruct()
        del _algoInstance
    return meta


def _metadata_streaming():
    meta = {}
    for name in skeys():
        essentia.log.debug(essentia.EPython, 'Loading __doc__ and __struct__ metadata for essentia.streaming class: %s' % name)
        _algoInstance = StreamingAlgorithm(name)
        meta[name] = {}
        meta[name]['__doc__'] = _algoInstance.getDoc()
        meta[name]['__struct__'] = _algoInstance.getStruct()
        del _algoInstance
    return meta


def _extract_metadata(filedir):
    """ Loads algorithms' metadata (__doc__ and __struct__) from the C extension
    and stores it to files in a filedir"""
    dump(_metadata_standard(), open(join(filedir, 'standard.meta.json'), 'w'))
    dump(_metadata_streaming(), open(join(filedir, 'streaming.meta.json'), 'w'))
