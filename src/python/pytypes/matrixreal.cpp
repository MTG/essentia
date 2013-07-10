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


DEFINE_PYTHON_TYPE(MatrixReal);


PyObject* MatrixReal::toPythonRef(TNT::Array2D<Real>* mat) {
  npy_intp dims[2];
  dims[0] = mat->dim1();
  dims[1] = mat->dim2();

  PyObject* result;
  if (dims[0] == 0 || dims[1] == 0) {
    result = PyArray_SimpleNew(2, dims, PyArray_FLOAT);
  }
  else {
    result = PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT, &((*mat)[0][0]));
  }

  if (result == NULL) {
    throw EssentiaException("MatrixReal: dang null object");
  }

  PyArray_BASE(result) = TO_PYTHON_PROXY(MatrixReal, mat);

  return result;
}


void* MatrixReal::fromPythonRef(PyObject* obj) {
  // Note! Even though the input numpy.array's BASE pointer might already be
  // pointing to a TNT::Array2D<Real> that we could just return right away, the
  // caller wouldn't know whether this was the case, or if a new TNT::Array2D
  // was created, and hence wouldn't know whether or not to call delete (they should not in the
  // first case, and they should in the second case). This problem is due to the
  // fact that we cannot create TNT::Array2D wrapper objects that just point to
  // a numpy.array's data and are ALWAYS safe to delete

  throw EssentiaException("MatrixReal::fromPythonRef: not implemented");
}


void* MatrixReal::fromPythonCopy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("MatrixReal::fromPythonRef: argument not a PyArray");
  }
  if (PyArray_NDIM(obj) != 2) {
    throw EssentiaException("MatrixReal::fromPythonRef: argument is not a 2-dimensional PyArray");
  }

  TNT::Array2D<Real>* tntmat = new TNT::Array2D<Real>(PyArray_DIM(obj, 0), PyArray_DIM(obj, 1), 0.0);

  // copy data from numpy array to matrix
  PyArrayObject* numpyarr = (PyArrayObject*)obj;

  for (int i=0; i<int(tntmat->dim1()); ++i) {
    const Real* src = (Real*)(numpyarr->data + i*numpyarr->strides[0]);
    Real* dest = &((*tntmat)[i][0]);
    fastcopy(dest, src, tntmat->dim2());
  }

  return tntmat;
}
