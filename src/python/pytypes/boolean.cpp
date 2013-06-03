#include "typedefs.h"
#include "parsing.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(Boolean);

PyObject* Boolean::toPythonCopy(const bool* x) {
  if (*x) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}


void* Boolean::fromPythonCopy(PyObject* obj) {
  if (!PyBool_Check(obj)) {
    throw EssentiaException("Boolean::fromPythonCopy: input is not a PyBool");
  }

  return new bool(obj == Py_True);
}

Parameter* Boolean::toParameter(PyObject* obj) {
  bool* value = (bool*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
