#include "typedefs.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(String);

PyObject* String::toPythonCopy(const string* s) {
  return PyString_FromStringAndSize(s->c_str(), s->size());
}


void* String::fromPythonCopy(PyObject* obj) {
  if (!PyString_Check(obj)) {
    throw EssentiaException("String::fromPythonCopy: input not a PyString: ", strtype(obj));
  }

  return new string(PyString_AS_STRING(obj));
}

Parameter* String::toParameter(PyObject* obj) {
  string* value = (string*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
