/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "pypool.h"
#include "parsing.h"
#include <iostream>
using namespace std;
using namespace essentia;


PyMethodDef PyPool_methods[] = {
  { "__add__",        (PyCFunction)PyPool::add, METH_VARARGS,
                      "Pool.add(key, value) adds \"value\" to the pool under \"key\"" },
  { "__set__",        (PyCFunction)PyPool::set, METH_VARARGS,
                      "Pool.set(key, value) sets \"value\" in the pool under \"key\"" },
  { "__merge__",      (PyCFunction)PyPool::merge, METH_VARARGS,
                      "Pool.merge(key, mergeType) merges \"value\" in the pool under \"key\" or "\
                       "Pool.merge(pool, mergeType) merges \"pool\" in the pool"},
  { "__mergeSingle__",(PyCFunction)PyPool::mergeSingle, METH_VARARGS,
                      "Pool.mergeSingle(key, value) sets \"value\" in the pool under \"key\"" },
  { "__value__",      (PyCFunction)PyPool::value, METH_VARARGS,
                      "Pool.value(key) retrieves a value from the pool under \"key\"" },
  { "isSingleValue",  (PyCFunction)PyPool::isSingleValue, METH_O,
                      "Pool.isSingleValue(key) returns true if the descriptor under \"key\" is a single value descriptor" },
  { "remove",         (PyCFunction)PyPool::remove, METH_O,
                      "Pool.remove(key) removes all values in the pool under \"key\"" },
  { "removeNamespace",(PyCFunction)PyPool::removeNamespace, METH_O,
                      "Pool.removeNamespace(namespace) removes all descriptors in the pool under \"namespace\"" },
  { "clear",         (PyCFunction)PyPool::clear, METH_NOARGS,
                      "Pool.clear() clears out the pool" },
  { "descriptorNames",(PyCFunction)PyPool::descriptorNames, METH_VARARGS,
                      "Pool.descriptorNames(namespace) returns a list of all descriptors in the pool under \"namespace\". If no namespace is supplied it returns a list of all descriptors" },
  { "__keyType__",    (PyCFunction)PyPool::keyType, METH_O,
                      "Returns the type of the data stored under 'key'" },
  { NULL }  /* Sentinel */
};


PyTypeObject PyPoolType = {
    PyObject_HEAD_INIT(NULL)
    0,                         // ob_size
    "essentia.Pool",           // tp_name
    sizeof(PyPool),            // tp_basicsize
    0,                         // tp_itemsize
    PyPool::dealloc,           // tp_dealloc
    0,                         // tp_print
    0,                         // tp_getattr
    0,                         // tp_setattr
    0,                         // tp_compare
    0,                         // tp_repr
    0,                         // tp_as_number
    0,                         // tp_as_sequence
    0,                         // tp_as_mapping
    0,                         // tp_hash
    0,                         // tp_call
    0,                         // tp_str
    0,                         // tp_getattro
    0,                         // tp_setattro
    0,                         // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Pool objects",            // tp_doc
    0,		               // tp_traverse
    0,		               // tp_clear
    0,		               // tp_richcompare
    0,		               // tp_weaklistoffset
    0,		               // tp_iter
    0,		               // tp_iternext
    PyPool_methods,                         // tp_methods
    0,                         // tp_members
    0,                         // tp_getset
    0,                         // tp_base
    0,                         // tp_dict
    0,                         // tp_descr_get
    0,                         // tp_descr_set
    0,                         // tp_dictoffset
    PyPool::init,              // tp_init
    0,                         // tp_alloc
    PyPool::make_new,          // tp_new
};


int PyPool::init(PyObject* self, PyObject* args, PyObject* kwds) {
  reinterpret_cast<PyPool*>(self)->pool = new essentia::Pool();

  // default constructor with no argument
  if (PyArg_ParseTuple(args, (char*)"")) return 0;

  return -1;
}


PyObject* PyPool::keyType(PyPool* self, PyObject* obj) {
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected a string argument");
    return NULL;
  }

  string key = PyString_AS_STRING(obj);
  Pool& p = *(self->pool);

  // search for the key and return the respective Edt of the sub-pool its in

  // search Real sub-pool
  if (p.getRealPool().find(key) != p.getRealPool().end()) {
    return PyString_FromString( edtToString(VECTOR_REAL).c_str() );
  }

  // search string sub-pool
  if (p.getStringPool().find(key) != p.getStringPool().end()) {
    return PyString_FromString( edtToString(VECTOR_STRING).c_str() );
  }

  // search StereoSample sub-pool
  if (p.getStereoSamplePool().find(key) != p.getStereoSamplePool().end()) {
    return PyString_FromString( edtToString(VECTOR_STEREOSAMPLE).c_str() );
  }

  // search vector<Real> sub-pool
  if (p.getVectorRealPool().find(key) != p.getVectorRealPool().end()) {
    return PyString_FromString( edtToString(VECTOR_VECTOR_REAL).c_str() );
  }

  // search vector<string> sub-pool
  if (p.getVectorStringPool().find(key) != p.getVectorStringPool().end()) {
    return PyString_FromString( edtToString(VECTOR_VECTOR_STRING).c_str() );
  }

  // search array2d sub-pool
  if (p.getArray2DRealPool().find(key) != p.getArray2DRealPool().end()) {
    return PyString_FromString( edtToString(VECTOR_MATRIX_REAL).c_str() );
  }

  // search single real pool
  if (p.getSingleRealPool().find(key) != p.getSingleRealPool().end()) {
    return PyString_FromString( edtToString(REAL).c_str() );
  }

  // search single vector real pool
  if (p.getSingleVectorRealPool().find(key) != p.getSingleVectorRealPool().end()) {
    return PyString_FromString( edtToString(VECTOR_REAL).c_str() );
  }

  // search single string pool
  if (p.getSingleStringPool().find(key) != p.getSingleStringPool().end()) {
    return PyString_FromString( edtToString(STRING).c_str() );
  }

  // couldn't find the key
  ostringstream msg;
  msg << "'" << key << "' is not a key in the pool" << endl;
  PyErr_SetString(PyExc_ValueError, msg.str().c_str());
  return NULL;
}

PyObject* PyPool::toPythonRef(Pool* data) {
  return TO_PYTHON_PROXY(PyPool, data);
}

Pool* PyPool::fromPythonRef(PyObject* obj) {
  if (!PyType_IsSubtype(obj->ob_type, &PyPoolType)) {
    throw EssentiaException("PyPool::fromPythonCopy: argument given is not a PyPool.\n"
                            "If it actually is one, also make sure your algorithm is among the ones being able to receive a pool as input (as defined in essentia/standard.py)");
  }

  return reinterpret_cast<PyPool*>(obj)->pool;
}


PyObject* PyPool::add(PyPool* self, PyObject* pyArgs) {
  vector<PyObject*> args = unpack(pyArgs);

  // make sure we have three args (key, type, value)
  if (args.size() != 4) {
    PyErr_SetString(PyExc_RuntimeError, "4 arguments required (string, string, value, bool)");
    return NULL;
  }

  // make sure first and second args are strings
  if (!PyString_Check(args[0]) || !PyString_Check(args[1])) {
    PyErr_SetString(PyExc_TypeError, "first argument should be a string");
    return NULL;
  }

  string key = PyString_AsString(args[0]);
  Edt tp = stringToEdt( PyString_AS_STRING(args[1]) );
  Pool& p = *(self->pool);
  if (!PyBool_Check(args[3])) {
    PyErr_SetString(PyExc_TypeError, "last argument should be a boolean");
  }

  bool validityCheck = (args[3]==Py_True);

  try {

    #define ADD_COPY(tname, type) { \
      type* val = (type*)tname::fromPythonCopy(args[2]); \
      p.add(key, *val, validityCheck); \
      delete val; \
      break; \
    }

    #define ADD_REF(tname, type) { \
      type* val = (type*)tname::fromPythonRef(args[2]); \
      p.add(key, *val, validityCheck); \
      delete val; \
      break; \
    }

    switch (tp) {
      case REAL:          ADD_COPY(PyReal, Real);
      case STRING:        ADD_COPY(String, string);
      case STEREOSAMPLE:  ADD_COPY(PyStereoSample, StereoSample);
      case VECTOR_STRING: ADD_COPY(VectorString, vector<string>);
      case MATRIX_REAL:   ADD_COPY(MatrixReal, TNT::Array2D<Real>);

      case VECTOR_REAL:   ADD_REF(VectorReal, RogueVector<Real>);

      default:
        ostringstream msg;
        msg << "Pool.add does not support the type: " << edtToString(tp);
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return NULL;
    }

    #undef ADD_COPY
    #undef ADD_REF
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "error while adding to the Pool: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject* PyPool::set(PyPool* self, PyObject* pyArgs) {
  vector<PyObject*> args = unpack(pyArgs);

  // make sure we have three args (key, type, value)
  if (args.size() != 4) {
    PyErr_SetString(PyExc_RuntimeError, "4 arguments required (string, string, value, bool)");
    return NULL;
  }

  // make sure first and second args are strings
  if (!PyString_Check(args[0]) || !PyString_Check(args[1])) {
    PyErr_SetString(PyExc_TypeError, "first argument should be a string");
    return NULL;
  }

  string key = PyString_AsString(args[0]);
  Edt tp = stringToEdt( PyString_AS_STRING(args[1]) );
  Pool& p = *(self->pool);
  if (!PyBool_Check(args[3])) {
    PyErr_SetString(PyExc_TypeError, "last argument should be a boolean");
  }

  bool validityCheck = (args[3]==Py_True);

  try {

    #define SET_COPY(tname, type) { \
      type* val = (type*)tname::fromPythonCopy(args[2]); \
      p.set(key, *val, validityCheck); \
      delete val; \
      break; \
    }

    #define SET_REF(tname, type) { \
      type* val = (type*)tname::fromPythonRef(args[2]); \
      p.set(key, *val, validityCheck); \
      delete val; \
      break; \
    }

    switch (tp) {
      case REAL: SET_COPY(PyReal, Real);
      case STRING: SET_COPY(String, string);
      case VECTOR_REAL: SET_REF(VectorReal, RogueVector<Real>);
      default:
        ostringstream msg;
        msg << "Pool.set does not support the type: " << edtToString(tp);
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return NULL;
    }

    #undef SET_COPY
    #undef SET_REF
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "error while setting value in the Pool: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject* PyPool::merge(PyPool* self, PyObject* pyArgs) {
  vector<PyObject*> args = unpack(pyArgs);

  // make sure we have three args (key, type, value)
  if (args.size() < 3) {
    PyErr_SetString(PyExc_RuntimeError, "4 arguments required (string, string, value, string)");
    return NULL;
  }

  // make sure first and second args are strings
  string key = "";
  Edt tp = UNDEFINED;
  string mergeType = "";
  Pool& p = *(self->pool);

  if (args.size() == 4) {
    if (!PyString_Check(args[0]) || !PyString_Check(args[1])) {
      PyErr_SetString(PyExc_TypeError, "pool.merge, first argument should be a string");
      return NULL;
    }
    if (!PyString_Check(args[3])) {
      PyErr_SetString(PyExc_TypeError, "pool.merge, 4rth argument should be a string");
      return NULL;
    }
    key = PyString_AsString(args[0]);
    tp = stringToEdt( PyString_AS_STRING(args[1]) );
    mergeType = PyString_AsString(args[3]);
  }
  else if(args.size() == 3) {
    tp = stringToEdt( PyString_AS_STRING(args[0]) );
    if (tp != POOL) {
      PyErr_SetString(PyExc_TypeError, "pool.merge, first argument should be a pool");
      return NULL;
    }
    if (!PyString_Check(args[2])) {
      PyErr_SetString(PyExc_TypeError, "pool.merge, 3rd argument should be a string");
      return NULL;
    }
    string mergeType = PyString_AsString(args[2]);
    try {
      p.merge(*PyPool::fromPythonRef(args[1]), mergeType);
      Py_RETURN_NONE;
    }
    catch (const exception& e) {
      ostringstream msg;
      msg << "pool.merge, error while merging two pools: " << e.what();
      PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
      return NULL;
    }
  }

  try {

    #define MERGE_COPY(tname, type) { \
      type* val = (type*)tname::fromPythonCopy(args[2]); \
      p.merge(key, *val, mergeType); \
      delete val; \
      break; \
    }

    #define MERGE_REF(tname, type) { \
      type* val = (type*)tname::fromPythonRef(args[2]); \
      p.merge(key, *val, mergeType); \
      delete val; \
      break; \
    }


    switch (tp) {
      // single types (not vectorized) are not possible types as they should
      // belong to single_valued_pools and not usual pools, hence they appear
      // in mergeSingle:
      case VECTOR_REAL:   MERGE_REF(VectorReal, RogueVector<Real>);
      case VECTOR_STRING: MERGE_COPY(VectorString, vector<string>);
      case VECTOR_STEREOSAMPLE: MERGE_COPY(VectorStereoSample, vector<StereoSample>);
      case VECTOR_VECTOR_REAL: MERGE_COPY(VectorVectorReal, vector<vector<Real> >);
      case VECTOR_VECTOR_STRING: MERGE_COPY(VectorVectorString, vector<vector<string> >);
      case VECTOR_MATRIX_REAL:   MERGE_COPY(VectorMatrixReal, vector<TNT::Array2D<Real> >);
      //case POOL: p.merge(args[1].cppPool, mergeType)


      default:
        ostringstream msg;
        msg << "Pool.merge does not support the type: " << edtToString(tp);
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return NULL;
    }

    #undef MERGE_COPY
    #undef MERGE_REF
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "Pool.merge error while merging into the Pool: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  Py_RETURN_NONE;
}

PyObject* PyPool::mergeSingle(PyPool* self, PyObject* pyArgs) {
  vector<PyObject*> args = unpack(pyArgs);

  // make sure we have three args (key, type, value)
  if (args.size() != 4) {
    PyErr_SetString(PyExc_RuntimeError, "4 arguments required (string, string, value, string)");
    return NULL;
  }

  // make sure first and second args are strings
  if (!PyString_Check(args[0]) || !PyString_Check(args[1])) {
    PyErr_SetString(PyExc_TypeError, "first argument should be a string");
    return NULL;
  }
  if (!PyString_Check(args[3])) {
    PyErr_SetString(PyExc_TypeError, "4th argument should be a string");
    return NULL;
  }

  string key = PyString_AsString(args[0]);
  Edt tp = stringToEdt( PyString_AS_STRING(args[1]) );
  Pool& p = *(self->pool);
  string mergeType = PyString_AsString(args[3]);

  try {

    #define MERGE_COPY(tname, type) { \
      type* val = (type*)tname::fromPythonCopy(args[2]); \
      p.mergeSingle(key, *val, mergeType); \
      delete val; \
      break; \
    }

    #define MERGE_REF(tname, type) { \
      type* val = (type*)tname::fromPythonRef(args[2]); \
      p.mergeSingle(key, *val, mergeType); \
      delete val; \
      break; \
    }

    switch (tp) {
      case REAL:        MERGE_COPY(PyReal, Real);
      case STRING:      MERGE_COPY(String, string);
      case VECTOR_REAL: MERGE_REF(VectorReal, RogueVector<Real>);
      default:
        ostringstream msg;
        msg << "Pool.mergeSingle does not support the type: " << edtToString(tp);
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return NULL;
    }

    #undef MERGE_COPY
    #undef MERGE_REF
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "Pool.mergeSingle error while merging value in the Pool: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  Py_RETURN_NONE;
}


PyObject* PyPool::value(PyPool* self, PyObject* pyArgs) {
  vector<PyObject*> args = unpack(pyArgs);

  // make sure we have two arg and they are strings
  if (args.size() != 2 || !PyString_Check(args[0]) || !PyString_Check(args[1])) {
    PyErr_SetString(PyExc_RuntimeError, "2 arguments required (string, string)");
    return NULL;
  }

  string key = PyString_AS_STRING(args[0]);
  Edt tp = stringToEdt( PyString_AS_STRING(args[1]) );
  Pool& p = *(self->pool);

  try {
    switch (tp) {
      case REAL: return PyReal::toPythonCopy(&p.value<Real>(key));
      case STRING: return String::toPythonCopy(&p.value<string>(key));
      case VECTOR_REAL: {
        // this is a special case, we can't create the RogueVector that wraps
        // the Pool's underlying std::vector because it might be the case that
        // the Pool is deleted and the numpy.array that points to the RogueVector
        // still exists
        const vector<Real>& v = p.value<vector<Real> >(key);
        RogueVector<Real>* r = new RogueVector<Real>(v.size(), 0.);
        for (int i=0; i<int(v.size()); ++i) { (*r)[i] = v[i]; }
        return VectorReal::toPythonRef(r);
      }
      case VECTOR_STRING: return VectorString::toPythonCopy(&p.value<vector<string> >(key));
      case VECTOR_STEREOSAMPLE: return VectorStereoSample::toPythonCopy(&p.value<vector<StereoSample> >(key));
      case VECTOR_VECTOR_REAL: return VectorVectorReal::toPythonCopy(&p.value<vector<vector<Real> > >(key));
      case VECTOR_VECTOR_STRING: return VectorVectorString::toPythonCopy(&p.value<vector<vector<string> > >(key));
      case VECTOR_MATRIX_REAL: return VectorMatrixReal::toPythonCopy(&p.value<vector<TNT::Array2D<Real> > >(key));
      default:
        ostringstream msg;
        msg << "Pool.value does not support the type: " << edtToString(tp);
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return NULL;
    }
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "error while retrieving value from Pool: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }
}


PyObject* PyPool::remove(PyPool* self, PyObject* obj) {
  // make sure first arg is a string
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expecting a string argument");
    return NULL;
  }

  self->pool->remove(PyString_AS_STRING(obj));
  Py_RETURN_NONE;
}

PyObject* PyPool::isSingleValue(PyPool* self, PyObject* obj) {
  // make sure first arg is a string
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expecting a string argument");
    return NULL;
  }

  if (self->pool->isSingleValue(PyString_AS_STRING(obj)))
    return Py_True;
  return Py_False;

}

PyObject* PyPool::removeNamespace(PyPool* self, PyObject* obj) {
  // make sure first arg is a string
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expecting a string argument");
    return NULL;
  }

  self->pool->removeNamespace(PyString_AS_STRING(obj));
  Py_RETURN_NONE;
}

PyObject* PyPool::descriptorNames(PyPool* self, PyObject* pyArgs) {

  vector<PyObject*> args = unpack(pyArgs);
  if (args.size() > 1) {
    PyErr_SetString(PyExc_TypeError, "expecting only one argument");
    return NULL;
  }
  if (args.size() == 0) {
    vector<string> dNames = self->pool->descriptorNames();
    return VectorString::toPythonCopy(&dNames);
  }
  if (!PyString_Check(args[0])) {
    PyErr_SetString(PyExc_TypeError, "expecting a string argument");
    return NULL;
  }
  vector<string> dNames = self->pool->descriptorNames(PyString_AS_STRING(args[0]));

  // convert dNames to python list
  return VectorString::toPythonCopy(&dNames);
}

PyObject* PyPool::clear(PyPool* self) {
  self->pool->clear();
  Py_RETURN_NONE;
}
