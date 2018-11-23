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

DEFINE_PYTHON_TYPE(VectorTensorReal);



PyObject* VectorTensorReal::toPythonCopy(const vector<Tensor<Real> >* aVec) {
  int size = aVec->size();
  PyObject* result = PyList_New(size);
  

  for (int i=0; i<size; ++i) {
    const TensorRef<Real> *a = &((*aVec)[i]);
    int dims = a->num_dimensions();
      npy_intp shape[dims];
      for (int j = 0; j< dims; j++)
          shape[j] = (long int)a->shape()[j];

    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(dims, shape, PyArray_FLOAT);
    if (mat == NULL) {
      throw EssentiaException("VectorTensorReal::toPythonCopy: dang null object");
    }
    // assert(result->strides[2] == sizeof(Real));

    // w/o the following check, will crash
    if (shape[1] != 0) {
      for (int j=0; j<(int)shape[0]; j++) {
        for (int k=0; k<(int)shape[1]; k++) {
          Real* dest = (Real*)(mat->data + j*mat->strides[0] + k*mat->strides[1]);
          const Real* src = &((*a)[j][k][0]);
          fastcopy(dest, src, shape[2]);
        }
      }
    }

    PyList_SET_ITEM(result, i, (PyObject*)mat);
  }

  return result;
}


void* VectorTensorReal::fromPythonCopy(PyObject* obj) {
  if (!PyList_Check(obj)) {
    throw EssentiaException("VectorTensorReal::fromPythonCopy: input is not a list");
  }

  int size = PyList_Size(obj);
  vector<Tensor<Real> >* v = new vector<Tensor<Real> >(size);

  for (int i=0; i<size; ++i) {
    try {
      (*v)[i] = *(Tensor<Real> *)TensorReal::fromPythonCopy(PyList_GET_ITEM(obj, i));
    }

    catch (const exception&) {
      delete v;
      throw;
    }

  }

  return v;
}
