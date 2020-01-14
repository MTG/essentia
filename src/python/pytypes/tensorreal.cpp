/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

DEFINE_PYTHON_TYPE(TensorReal);


PyObject* TensorReal::toPythonRef(essentia::Tensor<essentia::Real>* tensor) {
  PyObject* result;

  int nd = tensor->rank();
  npy_intp dims[nd];

  for (int i = 0; i < nd; i++)
    dims[i] = tensor->dimension(i);

  result = PyArray_SimpleNewFromData(nd, dims, PyArray_FLOAT, tensor->data());

  assert(result->strides[3] == sizeof(Real));

  if (result == NULL) {
    throw EssentiaException("TensorReal: dang null object");
  }

  PyArray_BASE(result) = TO_PYTHON_PROXY(TensorReal, tensor);
  return result;
}


void* TensorReal::fromPythonCopy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("TensorReal::fromPythonRef: expected PyArray, received: ", strtype(obj));
  }
  if (PyArray_NDIM(obj) != 4) {
    throw EssentiaException("TensorReal::fromPythonCopy: argument is not a 4-dimensional PyArray");
  }

  // // copy data from numpy array to matrix
  PyArrayObject* numpyarr = (PyArrayObject*)obj;
  if (numpyarr->descr->type_num != PyArray_FLOAT) {
     throw EssentiaException("TensorReal::fromPythonRef: this NumPy array doesn't contain Reals (maybe you forgot dtype='f4')");
   }

  return new Tensor<Real>(TensorMap<Real>((Real *)PyArray_DATA(numpyarr),
                                                  PyArray_DIM(obj, 0),
                                                  PyArray_DIM(obj, 1),
                                                  PyArray_DIM(obj, 2),
                                                  PyArray_DIM(obj, 3)));
}