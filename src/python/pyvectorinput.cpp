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

#include <Python.h>
#include "vectorinput.h"
#include "pytypes/pypool.h" // to use its type-determining capabilities

using namespace essentia;
using namespace std;

#define INIT_TYPE(CppType, initMethod) { \
  vector<CppType>* data = reinterpret_cast<vector<CppType>*>(initMethod(input)); \
  self->algo = reinterpret_cast<streaming::Algorithm*>(new streaming::VectorInput<CppType>(data)); \
  return 0; \
}

#define INIT_TYPE_OWNDATA(CppType, initMethod) { \
  vector<CppType>* data = reinterpret_cast<vector<CppType>*>(initMethod(input)); \
  self->algo = reinterpret_cast<streaming::Algorithm*>(new streaming::VectorInput<CppType>(data, true)); \
  return 0; \
}

static int vectorinput_init(PyStreamingAlgorithm* self, PyObject *args, PyObject *kwds) {
  vector<PyObject*> argsV = unpack(args);
  if (argsV.size() != 2) {
    PyErr_SetString(PyExc_ValueError, "VectorInput.__init__ requires 2 data arguments (value, type)");
    return -1;
  }

  PyObject* input = argsV[0];

  if (!PyString_Check(argsV[1])) {
    PyErr_SetString(PyExc_TypeError, "expecting second argument as string");
    return -1;
  }
  Edt tp = stringToEdt( PyString_AS_STRING(argsV[1]) );

  // NOTE:
  // Everywhere we use fromPythonCopy, we need to give ownership of that newly created variable to the
  // VectorInput instance, so that it knows it has to delete it later. This is done by using own=true in
  // the VectorInput constructor.
  try {
    switch (tp) {

    case VECTOR_REAL:    INIT_TYPE(Real, VectorReal::fromPythonRef);
    case VECTOR_INTEGER: INIT_TYPE(int, VectorInteger::fromPythonRef);

    case VECTOR_STRING:       INIT_TYPE_OWNDATA(string,             VectorString::fromPythonCopy);
    case VECTOR_STEREOSAMPLE: INIT_TYPE_OWNDATA(StereoSample,       VectorStereoSample::fromPythonCopy);
    case VECTOR_MATRIX_REAL:  INIT_TYPE_OWNDATA(TNT::Array2D<Real>, VectorMatrixReal::fromPythonCopy);
    case VECTOR_VECTOR_REAL:  INIT_TYPE_OWNDATA(vector<Real>,       VectorVectorReal::fromPythonCopy);

    case MATRIX_REAL: {
        TNT::Array2D<Real>* data = reinterpret_cast<TNT::Array2D<Real>*>(MatrixReal::fromPythonCopy(input));
        self->algo = reinterpret_cast<streaming::Algorithm*>(new streaming::VectorInput<vector<Real> >(*data));
        // VectorInput ctor with TNT::Array2D makes a copy of the data, so we need to delete it here
        delete data;
        return 0;
      }

    default:
        ostringstream msg;
        msg << "VectorInput does not yet support given type: " << edtToString(tp);
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
        return -1;
    }
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "An error occurred while creating VectorInput: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return -1;
  }
}

static PyTypeObject PyVectorInputType = {
  PyObject_HEAD_INIT(NULL)
  0,                                                      // ob_size
  "essentia.streaming.VectorInput",                       // tp_name
  sizeof(PyStreamingAlgorithm),                           // tp_basicsize
  0,                                                      // tp_itemsize
  PyStreamingAlgorithm::tp_dealloc,                       // tp_dealloc
  0,                                                      // tp_print
  0,                                                      // tp_getattr
  0,                                                      // tp_setattr
  0,                                                      // tp_compare
  0,                                                      // tp_repr
  0,                                                      // tp_as_number
  0,                                                      // tp_as_sequence
  0,                                                      // tp_as_mapping
  0,                                                      // tp_hash
  0,                                                      // tp_call
  0,                                                      // tp_str
  0,                                                      // tp_getattro
  0,                                                      // tp_setattro
  0,                                                      // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,               // tp_flags
  "essentia::streaming::VectorInput wrapper objects",     // tp_doc
  0,                                                      // tp_traverse
  0,                                                      // tp_clear
  0,                                                      // tp_richcompare
  0,                                                      // tp_weaklistoffset
  0,                                                      // tp_iter
  0,                                                      // tp_iternext
  PyStreamingAlgorithm_methods,                           // tp_methods
  0,                                                      // tp_members
  0,                                                      // tp_getset
  0,                                                      // tp_base
  0,                                                      // tp_dict
  0,                                                      // tp_descr_get
  0,                                                      // tp_descr_set
  0,                                                      // tp_dictoffset
  (initproc)vectorinput_init,                             // tp_init
  0,                                                      // tp_alloc
  PyStreamingAlgorithm::tp_new,                           // tp_new
};
