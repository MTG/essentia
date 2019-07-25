/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

  // check if all rows have the same size and convert to numpy array
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

  // convert to list of numpy arrays otherwise
  PyObject* result = PyList_New(v->size());

  for (int i=0; i<(int)v->size(); ++i) {
    npy_intp itemDims[1] = {(int)(*v)[i].size()};
    PyArrayObject* item = (PyArrayObject*)PyArray_SimpleNew(1, itemDims, PyArray_FLOAT);
    assert(item->strides[0] == sizeof(Real));
    if (item == NULL) {
      throw EssentiaException("VectorVectorReal: dang null object (list of numpy arrays)");
    }

    Real* dest = (Real*)(item->data);
    const Real* src = &((*v)[i][0]);
    fastcopy(dest, src, itemDims[0]);

    // old code to convert to list of lists
    /*
    PyObject* item = PyList_New((*v)[i].size());

    for (int j=0; j<(int)(*v)[i].size(); ++j) {
      double val = double((*v)[i][j]);
      PyList_SET_ITEM(item, j, PyFloat_FromDouble(val));
    }
    */

    PyList_SET_ITEM(result, i, (PyObject*) item);
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

    // List of floats
    if (PyList_Check(row)) {
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

    // Numpy array of floats
    else if (PyArray_Check(row)) {
      if (PyArray_NDIM(row) != 1) {
        throw EssentiaException("VectorVectorReal::fromPythonCopy: the element of input list "
                                "is not a 1-dimensional numpy array: ", PyArray_NDIM(row));
      }
      PyArrayObject* array = (PyArrayObject*)row;
      if (array == NULL) {
        throw EssentiaException("VectorVectorReal::fromPythonCopy: dang null object (list of numpy arrays)");
      }
      if (array->descr->type_num != PyArray_FLOAT) {
        throw EssentiaException("VectorVectorReal::fromPythonCopy: this NumPy array doesn't contain Reals (maybe you forgot dtype='f4')");
      }
      assert(array->strides[0] == sizeof(Real));
      npy_intp rowsize = array->dimensions[0];
      (*v)[i].resize(rowsize);
      const Real* src = (Real*)(array->data);
      Real* dest = &((*v)[i][0]);
      fastcopy(dest, src, rowsize);
    }

    // Unsupported
    else {
      delete v;
      throw EssentiaException("VectorVectorReal::fromPythonCopy: input is not a list of lists nor a list of numpy arrays");
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
