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
