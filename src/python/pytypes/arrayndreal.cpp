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

#include <numpy/arrayobject.h>
#include "typedefs.h"
#include "3rdparty/boost_1_68_0/boost/multi_array.hpp"
#include "3rdparty/boost_1_68_0/boost/python.hpp"
#include "3rdparty/boost_1_68_0/boost/cstdint.hpp"

using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(ArrayNDReal);

namespace detail {
  template<typename T>
  int numpy_type_map() {
    throw std::runtime_error("numpy_type_map(): Illegal conversion requested");
  }

  // Must be inlined to avoid multiple definitions since they are fully
  // specialized function templates.
  // template<> inline int numpy_type_map<Real>()                      { return NPY_FLOAT; }
}


template<class T, int NDims>
class numpy_boost : public boost::multi_array_ref<T, NDims>
{
public:
  typedef numpy_boost<T, NDims>            self_type;
  typedef boost::multi_array_ref<T, NDims> super;
  typedef typename super::size_type        size_type;
  typedef T*                               TPtr;

private:
  PyArrayObject* array;

  void init_from_array(PyArrayObject* a) throw() {
    /* Upon calling init_from_array, a should already have been
       incref'd for ownership by this object. */

    /* Store a reference to the Numpy array so we can DECREF it in the
       destructor. */
    array = a;

    /* Point the boost::array at the Numpy array data.
       We don't need to worry about free'ing this pointer, because it
       will always point to memory allocated as part of the data of a
       Numpy array.  That memory is managed by Python reference
       counting. */
    super::base_ = (TPtr)PyArray_DATA(a);

    /* Set the storage order.
       It would seem like we would want to choose C or Fortran
       ordering here based on the flags in the Numpy array.  However,
       those flags are purely informational, the actually information
       about storage order is recorded in the strides. */
    super::storage_ = boost::c_storage_order();

    /* Copy the dimensions from the Numpy array to the boost::array. */
    boost::detail::multi_array::copy_n(PyArray_DIMS(a), NDims, super::extent_list_.begin());

    /* Copy the strides from the Numpy array to the boost::array.
       Numpy strides are in bytes.  boost::array strides are in
       elements, so we need to divide. */
    for (size_t i = 0; i < NDims; ++i) {
      super::stride_list_[i] = PyArray_STRIDE(a, i) / sizeof(T);
    }

    /* index_base_list_ stores the bases of the indices in each
       dimension.  Since we want C-style and Numpy-style zero-based
       indexing, just fill it with zeros. */
    std::fill_n(super::index_base_list_.begin(), NDims, 0);

    /* We don't want any additional offsets.  If they exist, Numpy has
       already handled that for us when calculating the data pointer
       and strides. */
    super::origin_offset_ = 0;
    super::directional_offset_ = 0;

    /* Calculate the number of elements.  This has nothing to do with
       memory layout. */
    super::num_elements_ = std::accumulate(super::extent_list_.begin(),
                                           super::extent_list_.end(),
                                           size_type(1),
                                           std::multiplies<size_type>());
  }

public:
  /* Construct from an existing Numpy array */
  numpy_boost(PyArrayObject* a) :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {

    if (a == NULL) {
      throw boost::python::error_already_set();
    }

    init_from_array(a);
  }

  /* Copy constructor */
  numpy_boost(const self_type &other) throw() :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {
    Py_INCREF(other.array);
    init_from_array(other.array);
  }

  /* Construct a new array based on the given dimensions */
  // template<class ExtentsList>
  // explicit numpy_boost(const ExtentsList& extents) :
  //   super(NULL, std::vector<typename super::index>(NDims, 0)),
  //   array(NULL)
  // {
  //   npy_intp shape[NDims];
  //   PyArrayObject* a;

  //   boost::detail::multi_array::copy_n(extents, NDims, shape);

  //   a = (PyArrayObject*)PyArray_SimpleNew(
  //       NDims, shape, ::detail::numpy_type_map<T>());
  //   if (a == NULL) {
  //     throw boost::python::error_already_set();
  //   }

  //   init_from_array(a);
  // }

  /* Destructor */
  ~numpy_boost() {
    /* Dereference the numpy array. */
    Py_XDECREF(array);
  }

  /* Assignment operator */
  void operator=(const self_type &other) throw() {
    Py_INCREF(other.array);
    Py_DECREF(array);
    init_from_array(other.array);
  }

  /* Return the underlying Numpy array object.  [Borrowed
     reference] */
  PyObject*
  py_ptr() const throw() {
    return (PyObject*)array;
  }
};


// template <size_t n>
PyObject* ArrayNDReal::toPythonRef(ArrayND<Real, 3>* a) {
  npy_intp dim = a->num_dimensions();
  PyObject* result;

  if (dim > 0) result = PyArray_SimpleNewFromData(1, &dim, PyArray_FLOAT, &((*a)[0][0][0]));
  else         result = PyArray_SimpleNew(1, &dim, PyArray_FLOAT);

  if (result == NULL) {
    throw EssentiaException("ArrayNDReal: dang null object");
  }

  PyArray_BASE(result) = TO_PYTHON_PROXY(ArrayNDReal, a);

  return result;
}

void* ArrayNDReal::fromPythonRef(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw EssentiaException("ArrayNDReal::fromPythonRef: expected PyArray, received: ", strtype(obj));
  }

  PyArrayObject* array = (PyArrayObject*)obj;

  if (array->descr->type_num != PyArray_FLOAT) {
    throw EssentiaException("ArrayNDReal::fromPythonRef: this NumPy array doesn't contain Reals (maybe you forgot dtype='f4')");
  }
  // if (array->nd != 1) {
  //   throw EssentiaException("ArrayNDReal::fromPythonRef: this NumPy array has dimension ", array->nd, " (expected 1)");
  // }

  numpy_boost<Real, 3> ndarray(array);

  return new boost::multi_array_ref<Real, 3>(static_cast<boost::multi_array_ref<Real, 3>& >(ndarray));
}
