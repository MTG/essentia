#ifndef ESSENTIA_PYTHON_PYPOOL_H
#define ESSENTIA_PYTHON_PYPOOL_H

#include <Python.h>
#include "pool.h"
#include "typewrapper.h"

extern PyTypeObject PyPoolType;

// Note that some of the functionality of PyPool is defined directly in python,
// see src/python/essentia/common.py
// Also for algorithms that have a pool as input (i.e. yamlOutput, PCA...) they
// need to be included as exceptions in standard.py due to pypool need to be
// passed as pool.cppPool

class PyPool {
 public:
  PyObject_HEAD
  essentia::Pool* pool;

  BASIC_MEMORY_MANAGEMENT(PyPool, pool);

  static PyObject* make_new_from_data(PyTypeObject* type, PyObject* args,
                                      PyObject* kwds, essentia::Pool* data) {
    PyPool* self = (PyPool*)make_new(type, args, kwds);
    self->pool = data;
    return (PyObject*)self;
  }

  static int init(PyObject* self, PyObject* args, PyObject* kwds);

  static PyObject* toPythonRef(essentia::Pool* data);
  static essentia::Pool* fromPythonRef(PyObject* obj);
  static PyObject* add(PyPool* self, PyObject* pyArgs);
  static PyObject* set(PyPool* self, PyObject* pyArgs);
  static PyObject* merge(PyPool* self, PyObject* pyArgs);
  static PyObject* mergeSingle(PyPool* self, PyObject* pyArgs);
  static PyObject* value(PyPool* self, PyObject* pyArgs);
  static PyObject* getItem(PyPool* self, PyObject* key);
  static PyObject* isSingleValue(PyPool* self, PyObject* key);
  static PyObject* remove(PyPool* self, PyObject* pyArgs);
  static PyObject* removeNamespace(PyPool* self, PyObject* pyArgs);
  static PyObject* descriptorNames(PyPool* self, PyObject* pyArgs);
  static PyObject* clear(PyPool* self);
  static PyObject* keyType(PyPool* self, PyObject* obj);
};


#endif // ESSENTIA_PYTHON_PYPOOL_H
