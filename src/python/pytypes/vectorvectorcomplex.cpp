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

DEFINE_PYTHON_TYPE(VectorVectorComplex);

PyObject* VectorVectorComplex::toPythonCopy(const vector<vector<complex<Real> > >* v) {
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

    result = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_COMPLEX64);
    assert(result->strides[1] == sizeof(complex<Real>));

    if (result == NULL) {
      throw EssentiaException("VectorVectorComplex: dang null object");
    }

    for (int i=0; i<dims[0]; i++) {
      complex<Real>* dest = (complex<Real>*)(result->data + i*result->strides[0]);
      const complex<Real>* src = &((*v)[i][0]);
      fastcopy(dest, src, dims[1]);
    }

    return (PyObject*)result;
  }

  // added this to VectorVectorComplex could be made from unequal sizes
  PyObject* result = PyList_New(v->size());

  for (int i=0; i<(int)v->size(); ++i) {
    PyObject* item = PyList_New((*v)[i].size());

    for (int j=0; j<(int)(*v)[i].size(); ++j) {
      complex<Real> val = complex<Real>((*v)[i][j]);
      PyList_SET_ITEM(item, j, PyComplex_FromDoubles(double(val.real()), double(val.imag())));
    }

    PyList_SET_ITEM(result, i, item);
  }

  return result;
}


void* VectorVectorComplex::fromPythonCopy(PyObject* obj) {
  if (!PyList_Check(obj)) {
    throw EssentiaException("VectorVectorComplex::fromPythonCopy: input is not a list. Numpy vectors are not supported as input yet. Please cast it to Python list");
  }

  int size = PyList_Size(obj);
  vector<vector<complex<Real> > >* v = new vector<vector<complex<Real> > >(size, vector<complex<Real> >());

  for (int i=0; i<size; i++) {
    PyObject* row = PyList_GetItem(obj, i);
    if (!PyList_Check(obj)) {
      delete v;
      throw EssentiaException("VectorVectorComplex::fromPythonCopy: input is not a list of lists. Lists of Numpy vectors are not supported as input yet. Please cast it to Python list of lists");
    }

    int rowsize = PyList_Size(row);
    (*v)[i].resize(rowsize);

    for (int j=0; j<rowsize; j++) {
      PyObject* item = PyList_GetItem(row, j);
      try{
        Py_complex a =  PyComplex_AsCComplex(item);
        (*v)[i][j] = complex<Real>((Real) a.real, (Real) a.imag);
      }
      catch(const EssentiaException& e){
        ostringstream msg;
        msg << "VectorVectorComplex::fromPythonCopy: input is not a list of lists of complex" << e.what();
        delete v;
      }

      /*
      if (!PyComplex_Check(item)) {
        delete v;
        throw EssentiaException("VectorVectorComplex::fromPythonCopy: input is not a list of lists of complex");
      }
      (*v)[i][j] = complex<Real>(a.real, a.imag);
      */
    }
  }

  return v;
}
/*
Parameter* VectorVectorComplex::toParameter(PyObject* obj) {
  vector<vector<complex<Real> > >* value = (vector<vector<complex<Real> > >*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
*/
