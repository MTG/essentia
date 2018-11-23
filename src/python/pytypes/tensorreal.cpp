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
#include "numpyboost.h"

using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(TensorReal);


PyObject* TensorReal::toPythonCopy(const essentia::Tensor<essentia::Real>* a) {
  
  PyArrayObject* result;

  int dims = a->num_dimensions();
  
  npy_intp shape[dims];
  for (int i = 0; i< dims; i++)
      shape[i] = (long int)a->shape()[i];


  // TODO: this should be possible to di it in this way
  // if (dims > 0) result = PyArray_SimpleNewFromData(dims, shape, PyArray_FLOAT, a->origin());
  // else          result = PyArray_SimpleNew(1, shape, PyArray_FLOAT);

  result = (PyArrayObject*)PyArray_SimpleNew(dims, shape, PyArray_FLOAT);
  assert(result->strides[2] == sizeof(Real));
  for (int i=0; i<(int)shape[0]; i++) {
    for (int j=0; j<(int)shape[1]; j++) {
      Real* dest = (Real*)(result->data + i*result->strides[0] + j*result->strides[1]);
      const Real* src = &((*a)[i][j][0]);
      fastcopy(dest, src, shape[2]);
    }
  }

  if (result == NULL) {
    throw EssentiaException("TensorReal: dang null object");
  }

  return (PyObject*) result;
}

void* TensorReal::fromPythonCopy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("TensorReal::fromPythonCopy: expected PyArray, received: ", strtype(obj));
  }

  PyArrayObject* array = (PyArrayObject*)obj;

  if (array->descr->type_num != PyArray_FLOAT) {
    throw EssentiaException("TensorReal::fromPythonCopy: this NumPy array doesn't contain Reals (maybe you forgot dtype='f4')");
  }

  return new Tensor<Real>((TensorRef<Real>)NumpyBoost<Real, 3>(array));
}
