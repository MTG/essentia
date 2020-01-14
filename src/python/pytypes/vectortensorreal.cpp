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

DEFINE_PYTHON_TYPE(VectorTensorReal);



PyObject* VectorTensorReal::toPythonCopy(const vector<Tensor<Real> >* tenVec) {
  int size = tenVec->size();
  PyObject* result = PyList_New(size);

  for (int i=0; i<size; ++i) {
    const Tensor<Real>& tensor = (*tenVec)[i];

    int nd = tensor.rank();
    npy_intp dims[nd];

    for (int j = 0; j< nd; j++)
      dims[j] = tensor.dimension(j);

    PyArrayObject* numpyarr = (PyArrayObject*)PyArray_SimpleNew(nd, dims, PyArray_FLOAT);
    
    if (numpyarr == NULL) {
      throw EssentiaException("VectorTensorReal::toPythonCopy: dang null object");
    }

    Real* dest = (Real*)numpyarr->data;
    const Real* src = tensor.data();
    fastcopy(dest, src, tensor.size());

    PyList_SET_ITEM(result, i, (PyObject*)numpyarr);
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
