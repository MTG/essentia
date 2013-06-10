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
 * You should have received a copy of the GNU General Public License along with 
 * this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef ESSENTIA_CONFIG_H
#define ESSENTIA_CONFIG_H

#ifndef ESSENTIA_VERSION
#  define ESSENTIA_VERSION "Unknown"
#endif

/**
 *
 * Configuration file for compiling Essentia
 *
 * flags and \#define's that have effect on the compilation process
 *
 * - ESSENTIA_EXPORTS:
 *     if defined, essentia will be exporting symbols which are part of the main
 *     API. Use this when compiling it as a shared library.
 *
 * - STRIP_DOCUMENTATION:
 *     if defined, this will remove all documentation strings from the compiled
 *     library or executable
 *
 * - CASE_SENSITIVE:
 *     if defined, all string comparisons in essentia are case-sensitive,
 *     otherwise they are case-insensitive (slower).
 *
 * - SAFE_TYPE_COMPARISONS:
 *     when setting inputs/outputs of an algorithm, use safe type comparisons
 *     i.e.: based on the name of the type, rather than just on the pointer.
 *     This means that type comparisons will work across dynamic libraries
 *     boundaries, but is a little bit slower.
 *
 * - NO_DEFAULT_PARAMETERS:
 *     when configuring an algorithm, you need to specify all of the parameters
 *     when this option is define, otherwise essentia will throw an exception.
 *
 * - DEBUGGING_ENABLED:
 *     when defined, debugging will be enabled and you will be able to control
 *     it using the functions defined in the debugging.h file. It is recommended
 *     to disable it only in the situations where you need the absolute last drop
 *     of performance, because debug levels that are not activated do cost a
 *     little bit, but not that much really.
 */


#define ESSENTIA_EXPORTS
//#define STRIP_DOCUMENTATION
#define CASE_SENSITIVE
//#define SAFE_TYPE_COMPARISONS
//#define NO_DEFAULT_PARAMETERS
#define DEBUGGING_ENABLED



// tries to identify on which OS we are
#ifdef _MSC_VER
#  define OS_WIN32
#else
#  if defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__)
#    define OS_MAC
#  else
#    define OS_LINUX
#  endif
#endif


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
#ifdef ESSENTIA_EXPORTS
#  define ESSENTIA_API ESSENTIA_DLLEXPORT
#else
#  define ESSENTIA_API ESSENTIA_DLLIMPORT
#endif


// do we strip documentation or not?
#ifdef STRIP_DOCUMENTATION
#  define DOC(x) "Unavailable"
#else
#  define DOC(x) x
#endif


// case-insensitive or not?
#ifdef CASE_SENSITIVE
#  define string_cmp std::less<std::string>
#  define charptr_cmp strcmp
#else
#  define string_cmp case_insensitive_str_cmp
#  define charptr_cmp strcasecmp
#endif


#endif // ESSENTIA_CONFIG_H
