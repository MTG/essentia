#include "typedefs.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(VectorVectorString);

PyObject* VectorVectorString::toPythonCopy(const vector<vector<string> >* v) {
  PyObject* result = PyList_New(v->size());

  for (int i=0; i<int(v->size()); ++i) {
    PyObject* item = PyList_New((*v)[i].size());

    for (int j=0; j<int((*v)[i].size()); ++j) {
      const char* val = (*v)[i][j].c_str();
      PyList_SET_ITEM(item, j, PyString_FromString(val));
    }

    PyList_SET_ITEM(result, i, item);
  }

  return result;
}


void* VectorVectorString::fromPythonCopy(PyObject* obj) {
  if (!PyList_Check(obj)) {
    throw EssentiaException("VectorVectorString::fromPythonCopy: input not a PyList");
  }
  int size = PyList_Size(obj);
  vector<vector<string> >* v = new vector<vector<string> >(size);

  for (int i=0; i<size; ++i) {
    PyObject* row = PyList_GetItem(obj, i);
    if (!PyList_Check(obj)) {
      delete v;
      throw EssentiaException("VectorVectorString::fromPythonCopy: input not a PyList of PyLists");
    }

    int rowsize = PyList_Size(row);
    (*v)[i].resize(rowsize);

    for (int j=0; j<rowsize; ++j) {
      PyObject* item = PyList_GetItem(row, j);
      if (!PyString_Check(item)) {
        delete v;
        throw EssentiaException("VectorVectorString::fromPythonCopy: input not a PyList of PyLists of strings");
      }
      (*v)[i][j] = PyString_AsString(item);
    }
  }

  return v;
}

Parameter* VectorVectorString::toParameter(PyObject* obj) {
  vector<vector<string> >* value = (vector<vector<string> >*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
