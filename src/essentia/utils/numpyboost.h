/*
 * Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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


#ifndef NUMPYBOOST_H
#define NUMPYBOOST_H

#include <boost/multi_array.hpp>
#include <boost/python/detail/wrap_python.hpp>


namespace essentia {

namespace detail {
  template<typename T>
  int numpy_type_map() {
    throw std::runtime_error("numpy_type_map(): Illegal conversion requested");
  }

  // Must be inlined to avoid multiple definitions since they are fully
  // specialized function templates.
  // template<> inline int numpy_type_map<Real>()          { return NPY_FLOAT; }
}


template<class T, int NDims>
class NumpyBoost : public boost::multi_array_ref<T, NDims>
{
public:
  typedef NumpyBoost<T, NDims>            self_type;
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

    /* Copy the dimensions and strides from the Numpy array to the 
       boost::array. Numpy strides are in bytes. 
       This method accepts Numpy arrays of smaller dimension than the 
       final 4D tensor, {Batch, Channels, Timestamps, Feats}.
       The considered dimenension priority is, 
          - NDims = 2 -> Batch + Feats 
          - NDims = 3 -> Batch + Timestamps + Feats
          - NDims = 4 -> Batch + Channels + Timestamps + Feats */

    super::extent_list_[0] = PyArray_DIMS(a)[0];
    super::stride_list_[0] = PyArray_STRIDE(a, 0) / sizeof(T);

    /* Set singletion elements */
    for (size_t i = 1; i < NDims - PyArray_NDIM(a) + 1; i++) {
      super::extent_list_[i] = 1;
      super::stride_list_[i] = PyArray_STRIDE(a, 0) / sizeof(T);
    }

    /* Set elements from python */
    size_t idx = 1;
    for (size_t i = NDims - PyArray_NDIM(a) + 1; i < NDims; i++, idx++) {
      super::extent_list_[i] = PyArray_DIMS(a)[idx];
      super::stride_list_[i] = PyArray_STRIDE(a, idx) / sizeof(T);
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
  NumpyBoost(PyArrayObject* a) :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {

    init_from_array(a);
  }

  /* Copy constructor */
  NumpyBoost(const self_type &other) throw() :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {
    Py_INCREF(other.array);
    init_from_array(other.array);
  }

  /* Destructor */
  // ~NumpyBoost() {
    /* Dereference the numpy array. */
    //  Py_XDECREF(array);
  // }

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

} // namespace essentia

#endif // NUMPYBOOST_H