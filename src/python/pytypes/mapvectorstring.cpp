#include "typedefs.h"
using namespace std;
using namespace essentia;


DEFINE_PYTHON_TYPE(MapVectorString);


PyObject* MapVectorString::toPythonCopy(const map<string, vector<string> >* v) {
  throw EssentiaException("MapVectorString::fromPythonCopy currently not implemented");
}


void* MapVectorString::fromPythonCopy(PyObject* obj) {
  if (!PyDict_Check(obj)) {
    throw EssentiaException("MapVectorString::fromPythonCopy: expected PyDict, instead received: ", strtype(obj));
  }

  throw EssentiaException("MapVectorString::fromPythonCopy currently not implemented");
}

Parameter* MapVectorString::toParameter(PyObject* obj) {
  map<string, vector<string> >* value = (map<string, vector<string> >*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
