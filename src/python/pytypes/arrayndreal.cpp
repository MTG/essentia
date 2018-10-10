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

DEFINE_PYTHON_TYPE(ArrayNDReal);

// template <size_t n>
PyObject* ArrayNDReal::toPythonCopy(ArrayND<Real, 3>* a) {
  npy_intp dim = a->num_dimensions();
  PyObject* result;

  if (dim > 0) result = PyArray_SimpleNewFromData(1, &dim, PyArray_FLOAT, &((*a)[0]));
  else         result = PyArray_SimpleNew(1, &dim, PyArray_FLOAT);

  if (result == NULL) {
    throw EssentiaException("ArrayNDReal: dang null object");
  }

  PyArray_BASE(result) = TO_PYTHON_PROXY(ArrayND, a);

  return result;
}

void* ArrayNDReal::fromPythonCopy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("ArrayNDReal::fromPythonRef: expected PyArray, received: ", strtype(obj));
  }

  PyArrayObject* array = (PyArrayObject*)obj;

  if (array->descr->type_num != PyArray_FLOAT) {
    throw EssentiaException("ArrayNDReal::fromPythonRef: this NumPy array doesn't contain Reals (maybe you forgot dtype='f4')");
  }
  // if (array->nd != 1) {
  //   throw EssentiaException("ArrayNDReal::fromPythonRef: this NumPy array has dimension ", array->nd, " (expected 1)");
  // }
  size_t dims =  array->nd;

  npy_intp* shape = array->dimensions;


  return new boost::multi_array_ref<Real, 3>((Real*)PyArray_DATA(obj), boost::extents[shape[0]][shape[1]][shape[2]]);
}
