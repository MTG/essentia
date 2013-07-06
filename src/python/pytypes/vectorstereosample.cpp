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

DEFINE_PYTHON_TYPE(VectorStereoSample);

// converts a vector of size n to a numpy array of dimension n x 2
PyObject* VectorStereoSample::toPythonCopy(const vector<StereoSample>* v) {
  npy_intp dims[2] = {0, 0};
  dims[0] = v->size();
  dims[1] = 2;
  PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT);

  if (!result) throw EssentiaException("VectorStereoSample::toPythonCopy: could not create PyArray");

  for (int i=0; i<int(dims[0]); ++i) {
    Real* left = (Real*)(result->data + i*result->strides[0]);
    Real* right = (Real*)(result->data + i*result->strides[0] + result->strides[1]);
    *left = (*v)[i].left();
    *right = (*v)[i].right();
  }

  return (PyObject*)result;
}


void* VectorStereoSample::fromPythonCopy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("VectorStereoSample::fromPythonCopy: given input "
                            "is not a numpy array: ", strtype(obj));
  }

  if (PyArray_NDIM(obj) != 2) {
    throw EssentiaException("VectorStereoSample::fromPythonCopy: given input "
                            "is not a 2-dimensional numpy array: ", PyArray_NDIM(obj));
  }

  if (PyArray_DIM(obj, 1) != 2) {
    throw EssentiaException("VectorStereoSample::fromPythonCopy: given input's "
                            "second dimension is not 2: ", PyArray_DIM(obj, 1));
  }

  Py_ssize_t total = PyArray_DIM(obj, 0);
  PyArrayObject* arr = (PyArrayObject*)obj;
  vector<StereoSample>* result = new vector<StereoSample>(total);

  for (int i=0; i<int(total); ++i) {
    Real* left = (Real*)(arr->data + i*arr->strides[0]);
    Real* right = (Real*)(arr->data + i*arr->strides[0] + arr->strides[1]);
    (*result)[i].left() = *left;
    (*result)[i].right() = *right;
  }

  return result;
}
