#include "typedefs.h"
#include "parsing.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(PyStereoSample);

PyObject* PyStereoSample::toPythonCopy(const StereoSample* x) {
  PyObject* pyx = PyTuple_Pack(2, PyFloat_FromDouble(x->left()),
                                  PyFloat_FromDouble(x->right()));

  if (pyx == NULL) {
    throw EssentiaException("PyStereoSample::toPythonCopy: could not create tuple");
  }

  return pyx;
}


void* PyStereoSample::fromPythonCopy(PyObject* obj) {
  if (!PyTuple_Check(obj)) {
    throw EssentiaException("PyStereoSample::fromPythonCopy: input not a tuple: ", strtype(obj));
  }

  if (PyTuple_GET_SIZE(obj) != 2) {
    throw EssentiaException("PyStereoSample::fromPythonCopy: input tuple is not of size 2: ", PyTuple_GET_SIZE(obj));
  }

  // extract stereo values
  Real* left = reinterpret_cast<Real*>(PyReal::fromPythonCopy( PyTuple_GET_ITEM(obj, 0) ));
  Real* right = reinterpret_cast<Real*>(PyReal::fromPythonCopy( PyTuple_GET_ITEM(obj, 1) ));

  StereoSample* ss = new StereoSample();
  ss->left() = *left;
  ss->right() = *right;

  delete left;
  delete right;

  return ss;
}
