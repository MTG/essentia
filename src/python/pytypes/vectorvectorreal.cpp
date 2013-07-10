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

#include "typedefs.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(VectorVectorReal);

PyObject* VectorVectorReal::toPythonCopy(const vector<vector<Real> >* v) {
  npy_intp dims[2] = { 0, 0 };
  dims[0] = v->size();
  if (!v->empty()) dims[1] = (*v)[0].size();

  bool isRectangular = true;

  // check all rows have the same size
  for (int i=1; i<dims[0]; i++) {
    if ((int)(*v)[i].size() != dims[1]) {
      isRectangular = false;
    }
  }

  if (isRectangular && dims[0] > 0 && dims[1] > 0) {
    PyArrayObject* result;

    result = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_FLOAT);
    assert(result->strides[1] == sizeof(Real));

    if (result == NULL) {
      throw EssentiaException("VectorVectorReal: dang null object");
    }

    for (int i=0; i<dims[0]; i++) {
      Real* dest = (Real*)(result->data + i*result->strides[0]);
      const Real* src = &((*v)[i][0]);
      fastcopy(dest, src, dims[1]);
    }

    return (PyObject*)result;
  }

  // added this to vectorvectorreal could be made from unequal sizes
  PyObject* result = PyList_New(v->size());

  for (int i=0; i<(int)v->size(); ++i) {
    PyObject* item = PyList_New((*v)[i].size());

    for (int j=0; j<(int)(*v)[i].size(); ++j) {
      double val = double((*v)[i][j]);
      PyList_SET_ITEM(item, j, PyFloat_FromDouble(val));
    }

    PyList_SET_ITEM(result, i, item);
  }

  return result;
}


void* VectorVectorReal::fromPythonCopy(PyObject* obj) {
  if (!PyList_Check(obj)) {
    throw EssentiaException("VectorVectorReal::fromPythonCopy: input is not a list");
  }

  int size = PyList_Size(obj);
  vector<vector<Real> >* v = new vector<vector<Real> >(size, vector<Real>());

  for (int i=0; i<size; i++) {
    PyObject* row = PyList_GetItem(obj, i);
    if (!PyList_Check(obj)) {
      delete v;
      throw EssentiaException("VectorVectorReal::fromPythonCopy: input is not a list of lists");
    }

    int rowsize = PyList_Size(row);
    (*v)[i].resize(rowsize);

    for (int j=0; j<rowsize; j++) {
      PyObject* item = PyList_GetItem(row, j);
      if (!PyFloat_Check(item)) {
        delete v;
        throw EssentiaException("VectorVectorReal::fromPythonCopy: input is not a list of lists of floats");
      }
      (*v)[i][j] = PyFloat_AsDouble(item);
    }
  }

  return v;
}

Parameter* VectorVectorReal::toParameter(PyObject* obj) {
  vector<vector<Real> >* value = (vector<vector<Real> >*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
