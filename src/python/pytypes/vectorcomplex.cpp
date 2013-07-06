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

DEFINE_PYTHON_TYPE(VectorComplex);


PyObject* VectorComplex::toPythonRef(RogueVector<complex<Real> >* v) {
  npy_intp dim = v->size();
  PyObject* result;

  if (dim > 0) result = PyArray_SimpleNewFromData(1, &dim, PyArray_COMPLEX64, &((*v)[0]));
  else         result = PyArray_SimpleNew(1, &dim, PyArray_COMPLEX64);

  if (result == NULL) {
    throw EssentiaException("VectorComplex::toPythonRef: could not create PyArray of type PyArray_COMPLEX64");
  }

  // set the PyArray pointer to our vector, so it can be released when the
  // PyArray is released
  PyArray_BASE(result) = TO_PYTHON_PROXY(VectorComplex, v);

  return result;
}


void* VectorComplex::fromPythonRef(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("VectorComplex::fromPythonRef: input not a PyArray");
  }

  PyArrayObject* array = (PyArrayObject*)obj;

  if (array->descr->type_num != PyArray_CFLOAT) {
    throw EssentiaException("VectorComplex::fromPythonRef: this NumPy array doesn't contain complex<Real> (maybe you forgot dtype='c8')");
  }
  if (array->nd != 1) {
    throw EssentiaException("VectorComplex::fromPythonRef: this NumPy array has dimension ", array->nd, " (expected 1)");
  }

  return new RogueVector<complex<Real> >((complex<Real>*)PyArray_DATA(obj), PyArray_SIZE(obj));
}
