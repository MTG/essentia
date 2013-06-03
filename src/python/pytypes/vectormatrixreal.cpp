#include "typedefs.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(VectorMatrixReal);


PyObject* VectorMatrixReal::toPythonCopy(const vector<TNT::Array2D<Real> >* matVec) {
  int size = matVec->size();
  PyObject* result = PyList_New(size);

  for (int i=0; i<size; ++i) {
    npy_intp dims[2] = { 0, 0 };
    dims[0] = (*matVec)[i].dim1();
    dims[1] = (*matVec)[i].dim2();

    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_FLOAT);

    if (mat == NULL) {
      throw EssentiaException("VectorMatrixReal::toPythonCopy: dang null object");
    }

    // w/o the following check, will crash
    if (dims[1] != 0) {
      for (int j=0; j<dims[0]; ++j) {
        Real* dest = (Real*)(mat->data + j*mat->strides[0]);
        const Real* src = &((*matVec)[i][j][0]);
        fastcopy(dest, src, dims[1]);
      }
    }

    PyList_SET_ITEM(result, i, (PyObject*)mat);
  }

  return result;
}


void* VectorMatrixReal::fromPythonCopy(PyObject* obj) {
  if (!PyList_Check(obj)) {
    throw EssentiaException("VectorMatrixReal::fromPythonCopy: input is not a list");
  }

  int size = PyList_Size(obj);
  vector<TNT::Array2D<Real> >* v = new vector<TNT::Array2D<Real> >(size);

  for (int i=0; i<size; ++i) {
    TNT::Array2D<Real>* mat;
    try {
      mat = reinterpret_cast<TNT::Array2D<Real>*>(MatrixReal::fromPythonCopy(PyList_GET_ITEM(obj, i)));
    }
    catch (const exception&) {
      delete v;
      throw;
    }

    (*v)[i] = mat->copy();
  }

  return v;
}
