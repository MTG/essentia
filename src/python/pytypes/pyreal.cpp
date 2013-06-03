#include "typedefs.h"
#include "parsing.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(PyReal);

PyObject* PyReal::toPythonCopy(const Real* x) {
  return PyFloat_FromDouble(*x);
}


void* PyReal::fromPythonCopy(PyObject* obj) {
  if (!(PyFloat_Check(obj) || PyInt_Check(obj))) {
    throw EssentiaException("PyReal::fromPythonCopy: given value is not a float or int: ", strtype(obj));
  }

  return new Real(PyFloat_AsDouble(obj));
}

Parameter* PyReal::toParameter(PyObject* obj) {
  Real* value = (Real*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
