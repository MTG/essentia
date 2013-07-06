/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_PYTHON_TYPEWRAPPER_H
#define ESSENTIA_PYTHON_TYPEWRAPPER_H


#define BASIC_MEMORY_MANAGEMENT(className, variable)                           \
  static PyObject* make_new(PyTypeObject* type, PyObject* args, PyObject* kwds) { \
    return (PyObject*)(type->tp_alloc(type, 0));                               \
  }                                                                            \
                                                                               \
  static void dealloc(PyObject* self) {                                        \
    /*std::cout << "deallocing: data: " << reinterpret_cast<className*>(self)->variable << std::endl; \
    std::cout << "deallocing: self: " << self << std::endl;*/                    \
    delete reinterpret_cast<className*>(self)->variable;                       \
    /*std::cout << "successfully dealloced data" << std::endl;*/                   \
    reinterpret_cast<className*>(self)->variable = NULL;                       \
    self->ob_type->tp_free((PyObject*)self);                                   \
    /*std::cout << "successfully dealloced self" << std::endl;*/                   \
  }


#define DEFAULT_MEMORY_MANAGEMENT(className, wrappedType, variable)            \
  BASIC_MEMORY_MANAGEMENT(className, variable);                                \
                                                                               \
  static PyObject* make_new_from_data(PyTypeObject* type, PyObject* args,      \
                                      PyObject* kwds, wrappedType* data) {     \
    className* self = (className*)make_new(type, args, kwds);                  \
    self->variable = data;                                                     \
    return (PyObject*)self;                                                    \
  }                                                                            \
                                                                               \
  static int init(PyObject* self, PyObject* args, PyObject* kwds) {            \
    if (!PyArg_ParseTuple(args, (char*)"")) return -1;                         \
    return 0;                                                                  \
  }


#define DECLARE_PROXY_TYPE(className, type)                                    \
class className {                                                              \
 public:                                                                       \
  PyObject_HEAD                                                                \
  type* data;                                                                  \
                                                                               \
  DEFAULT_MEMORY_MANAGEMENT(className, type, data);                            \
                                                                               \
  static PyObject* toPythonRef(type* data);                                    \
  static PyObject* toPythonCopy(const type* data);                             \
  static void* fromPythonRef(PyObject* obj);                                   \
  static void* fromPythonCopy(PyObject* obj);                                  \
  static essentia::Parameter* toParameter(PyObject* obj);                      \
}


#define TO_PYTHON_PROXY(className, data) make_new_from_data(&className##Type, NULL, NULL, data)

#define DECLARE_PYTHON_TYPE(type) \
extern PyTypeObject type##Type;

#define DEFINE_PYTHON_TYPE(type)                                               \
PyTypeObject type##Type = {                                                    \
  PyObject_HEAD_INIT(NULL)                                                     \
  0,                                        /* ob_size           */            \
  "essentia." #type,                        /* tp_name           */            \
  sizeof(type),                             /* tp_basicsize      */            \
  0,                                        /* tp_itemsize       */            \
  type::dealloc,                            /* tp_dealloc        */            \
  0,                                        /* tp_print          */            \
  0,                                        /* tp_getattr        */            \
  0,                                        /* tp_setattr        */            \
  0,                                        /* tp_compare        */            \
  0,                                        /* tp_repr           */            \
  0,                                        /* tp_as_number      */            \
  0,                                        /* tp_as_sequence    */            \
  0,                                        /* tp_as_mapping     */            \
  0,                                        /* tp_hash           */            \
  0,                                        /* tp_call           */            \
  0,                                        /* tp_str            */            \
  0,                                        /* tp_getattro       */            \
  0,                                        /* tp_setattro       */            \
  0,                                        /* tp_as_buffer      */            \
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags          */            \
  #type " objects",                         /* tp_doc            */            \
  0,		                                /* tp_traverse       */            \
  0,		                                /* tp_clear          */            \
  0,		                                /* tp_richcompare    */            \
  0,		                                /* tp_weaklistoffset */            \
  0,		                                /* tp_iter           */            \
  0,		                                /* tp_iternext       */            \
  0,                                        /* tp_methods        */            \
  0,                                        /* tp_members        */            \
  0,                                        /* tp_getset         */            \
  0,                                        /* tp_base           */            \
  0,                                        /* tp_dict           */            \
  0,                                        /* tp_descr_get      */            \
  0,                                        /* tp_descr_set      */            \
  0,                                        /* tp_dictoffset     */            \
  type::init,                               /* tp_init           */            \
  0,                                        /* tp_alloc          */            \
  type::make_new,                           /* tp_new            */            \
}


#endif // ESSENTIA_PYTHON_TYPEWRAPPER_H
