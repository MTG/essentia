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

#ifndef ESSENTIA_CONFIG_H
#define ESSENTIA_CONFIG_H

/**
 * Essentia version number.
 */
#ifndef ESSENTIA_VERSION
#define ESSENTIA_VERSION "2.0.1"
#endif


/**
 * if set to @c 1, debugging will be enabled and you will be able to control
 * it using the functions defined in the debugging.h file. It is recommended
 * to disable it only in the situations where you need the absolute last drop
 * of performance, because debug levels that are not activated do cost a
 * little bit, but not that much really.
 */
#ifndef DEBUGGING_ENABLED
#define DEBUGGING_ENABLED 1
#endif


/**
 * if set to @c 1, essentia will be exporting symbols which are part of the main
 * API. Use this when compiling it as a shared library. Also needed for
 * compiling the python bindings.
 */
#ifndef ESSENTIA_EXPORTS
#define ESSENTIA_EXPORTS 1
#endif


/**
 * if set to @c 1, this will remove all documentation strings from the compiled
 * library or executable.
 */
#ifndef STRIP_DOCUMENTATION
#define STRIP_DOCUMENTATION 0
#endif


/**
 * if set to @c 1, all string comparisons in essentia are case-sensitive,
 * otherwise they are case-insensitive (slower).
 */
#ifndef CASE_SENSITIVE
#define CASE_SENSITIVE 1
#endif


/**
 *  if set to @c 1, essentia will use safe type comparisons (i.e.: based on
 *  the name of the type, rather than just on the pointer) when setting
 *  inputs/outputs of an algorithm. This allows type comparisons to work across
 *  dynamic libraries boundaries, but is a little bit slower.
 */
#ifndef SAFE_TYPE_COMPARISONS
#define SAFE_TYPE_COMPARISONS 0
#endif


/**
 * - if set to @c 1, this allows to use default values for the parameters when
 *   the algorithm defines them.
 * - if set to @c 0, this requires algorithm instantiations to always explicitly
 *   specify all the parameters. If they are not all specified, the
 *   instantiation of the algorithm will fail and throw an exception.
 */
#ifndef ALLOW_DEFAULT_PARAMETERS
#define ALLOW_DEFAULT_PARAMETERS 1
#endif



/**
 * OS type.
 */
#ifdef _MSC_VER
#  define OS_WIN32
#else
#  if defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__)
#    define OS_MAC
#  else
#    define OS_LINUX
#  endif
#endif


#ifndef DOXYGEN_SHOULD_SKIP_THIS

// some Windows peculiarities that need to be fixed
#ifdef OS_WIN32

  #pragma warning (disable : 4251 4275) // disable the DLL warnings...
  #pragma warning (disable : 4244 4305 4267) // disable float<=>double conversion warnings
  #pragma warning (disable : 4996) // XYZ was declared deprecated
  #pragma warning (disable : 4146) // MersenneTwister.h:273 unary minus operator applied to unsigned type, result still unsigned
  #pragma warning (disable : 4355) // this used in class initialization, but we do it in a safe way

  // tell microsoft we would like to use std::min and std::max
  #define NOMINMAX

  typedef unsigned int uint;

  #define strcasecmp _stricmp

  #include <float.h>

  namespace std {
    template <typename T>
    inline bool isnan(T x) {
      return _isnan(x) != 0;
    }
    template <typename T>
    inline bool isinf(T x) {
      return _finite(x) == 0;
    }
  }
#endif // OS_WIN32


#ifdef OS_MAC
typedef unsigned int uint;
#endif // OS_MAC


// returns GCC version as a single constant, valid for both linux & mac
#ifndef OS_WIN32
#  define GCC_VERSION (__GNUC__ * 10000     \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)
#endif // OS_WIN32


// Visibility options.
// Functions will not appear as part of a DSO exported objects by default
// unless they have been marked with an ESSENTIA_API qualifier.
// See http://gcc.gnu.org/wiki/Visibility for more information on the subject.
#ifdef OS_WIN32
#  define ESSENTIA_DLLEXPORT __declspec(dllexport)
#  define ESSENTIA_DLLIMPORT __declspec(dllimport)
#else
#  if (GCC_VERSION >= 40000)
#    define ESSENTIA_DLLEXPORT __attribute__ ((visibility("default")))
#    define ESSENTIA_DLLIMPORT __attribute__ ((visibility("default")))
#  else
#    define ESSENTIA_DLLEXPORT
#    define ESSENTIA_DLLIMPORT
#  endif
#endif


// define ESSENTIA_EXPORTS when building the DLL, don't when importing it
// into your application.
#if ESSENTIA_EXPORTS
#  define ESSENTIA_API ESSENTIA_DLLEXPORT
#else
#  define ESSENTIA_API ESSENTIA_DLLIMPORT
#endif


// do we strip algorithm documentation from the resulting binary or not?
#if STRIP_DOCUMENTATION
#  define DOC(x) "Unavailable"
#else
#  define DOC(x) x
#endif


// case-insensitive or not?
#if CASE_SENSITIVE
#  define string_cmp std::less<std::string>
#  define charptr_cmp strcmp
#else
#  define string_cmp case_insensitive_str_cmp
#  define charptr_cmp strcasecmp
#endif


#endif // DOXYGEN_SHOULD_SKIP_THIS


#endif // ESSENTIA_CONFIG_H
