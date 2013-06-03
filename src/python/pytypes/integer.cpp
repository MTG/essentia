#include "typedefs.h"
#include "parsing.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(Integer);

PyObject* Integer::toPythonCopy(const int* x) {
  return PyInt_FromLong(*x);
}


void* Integer::fromPythonCopy(PyObject* obj) {
  if (!PyInt_Check(obj)) {
    throw EssentiaException("Integer::fromPythonCopy: input is not a PyInt");
  }

  return new int(PyInt_AsLong(obj));
}

Parameter* Integer::toParameter(PyObject* obj) {
  int* value = (int*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
