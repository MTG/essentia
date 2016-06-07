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

#ifndef ESSENTIA_TYPES_H
#define ESSENTIA_TYPES_H

#include <map>
#include <vector>
#include <cctype>
#include <cassert>
#include <sstream>
#include <typeinfo>
#include <string.h>
#include "config.h"
#include "debugging.h"
#include "streamutil.h"


// fixed-size int types

#ifndef OS_WIN32

#include <inttypes.h>

typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int16_t sint16;
typedef int32_t sint32;
typedef int64_t sint64;

typedef unsigned int uint;

#else // OS_WIN32

typedef unsigned __int16 uint16;
typedef unsigned __int32 uint32;
typedef unsigned __int64 uint64;
typedef __int16 sint16;
typedef __int32 sint32;
typedef __int64 sint64;

#endif // OS_WIN32



namespace essentia {

/**
 * The main typedef for real numbers.
 */
typedef float Real;


/**
 * Exception class for Essentia. It has a whole slew of different constructors
 * to make it as easy as possible to throw an exception with a descriptive
 * message.
 */
class EssentiaException : public std::exception {

 public:
  EssentiaException(const char* msg) : exception(), _msg(msg) {}
  EssentiaException(const std::string& msg) : exception(), _msg(msg) {}
  EssentiaException(const std::ostringstream& msg) : exception(), _msg(msg.str()) {}

  template <typename T, typename U>
  EssentiaException(const T& a, const U& b) : exception() {
    std::ostringstream oss; oss << a << b; _msg = oss.str();
  }

  template <typename T, typename U, typename V>
  EssentiaException(const T& a, const U& b, const V& c) : exception() {
    std::ostringstream oss; oss << a << b << c; _msg = oss.str();
  }

  template <typename T, typename U, typename V, typename W>
  EssentiaException(const T& a, const U& b, const V& c, const W& d) : exception() {
    std::ostringstream oss; oss << a << b << c << d; _msg = oss.str();
  }

  virtual ~EssentiaException() throw() {}
  virtual const char* what() const throw() { return _msg.c_str(); }

 protected:
  std::string _msg;

};


/**
 * Case-insensitive compare function for characters.
 */
inline bool case_insensitive_char_cmp(char a, char b) {
  return std::tolower(a) < std::tolower(b);
}

/**
 * Function object for comparing two strings in a case-insensitive manner.
 */
struct case_insensitive_str_cmp
  : public std::binary_function<const std::string&, const std::string&, bool> {
  bool operator()(const std::string& str1, const std::string& str2) const {
    return std::lexicographical_compare(str1.begin(), str1.end(),
                                        str2.begin(), str2.end(),
                                        case_insensitive_char_cmp);
  }
};


template <class T>
class OrderedMap : public std::vector<std::pair<std::string, T*> > {
 public:
  typedef typename std::vector<std::pair<std::string, T*> > BaseClass;

  int size() const { return (int)BaseClass::size(); }

  const std::pair<std::string, T*>& operator[](uint idx) const {
    return BaseClass::operator[](idx);
  }

  std::pair<std::string, T*>& operator[](uint idx) {
    return BaseClass::operator[](idx);
  }

  const T* operator[](const char* str) const {
    const uint size = this->size();
    for (uint i=0; i<size; i++) {
      if (charptr_cmp((*this)[i].first.c_str(), str) == 0) {
        return (*this)[i].second;
      }
    }

    throw EssentiaException("Value not found: '", str, "'\nAvailable keys: ", keys());
  }

  T* operator[](const char* str) {
    return const_cast<T*>(const_cast<const OrderedMap<T>*>(this)->operator[](str));
  }

  const T* operator[](const std::string& str) const {
    return operator[](str.c_str());
  }

  T* operator[](const std::string& str) {
    return operator[](str.c_str());
  }

  std::vector<std::string> keys() const {
    std::vector<std::string> result(this->size());
    for (int i=0; i<this->size(); i++) {
      result[i] = this->at(i).first;
    }
    return result;
  }

  void insert(const std::string& key, T* value) {
    this->push_back(make_pair(key, value));
  }
};



/**
 * Special version of a std::map that allows us to use the [] operator on a
 * const object. In this case, if the key is found, it returns the associated
 * value, otherwise it throws an exception.
 * If not used on a constant object, it also throws an exception if the key is
 * not found, in order to have a consistent behavior.
 * It also redefines the insert() method to be more convenient.
 */
template <typename KeyType, typename ValueType, typename Compare = std::less<KeyType> >
class EssentiaMap : public std::map<KeyType, ValueType, Compare> {

 public:
  typedef std::map<KeyType, ValueType, Compare> BaseClass;

  /**
   * Classic version of the map accessor.
   */
  ValueType& operator[](const KeyType& key) {
    typename BaseClass::iterator it = this->find(key);
    if (it == BaseClass::end()) {
      throw EssentiaException("Value not found: '", key, "'\nAvailable keys: ", keys());
    }
    return it->second;
  }

  /**
   * New version that can be called on a constant object and instead of
   * creating a new null object and inserting it in the map just throws an
   * exception.
   */
  const ValueType& operator[](const KeyType& key) const {
    typename BaseClass::const_iterator it = this->find(key);
    if (it == BaseClass::end()) {
      throw EssentiaException("Value not found: '", key, "'\nAvailable keys: ", keys());
    }
    return it->second;
  }

  std::pair<typename BaseClass::iterator, bool> insert(const KeyType& key, const ValueType& value) {
    return BaseClass::insert(std::make_pair(key, value));
  }

  std::vector<std::string> keys() const {
    std::vector<std::string> result;
    result.reserve(BaseClass::size());
    std::ostringstream stream;
    typename BaseClass::const_iterator it = this->begin();
    for (; it != this->end(); ++it) {
      stream.str("");
      stream << it->first;
      result.push_back(stream.str());
    }
    return result;
  }

};


/**
 * Type of map used for storing the description of the various fields.
 */
typedef EssentiaMap<std::string, std::string, string_cmp> DescriptionMap;




/**
 * Return @c true if the two given types are the same.
 */
#if SAFE_TYPE_COMPARISONS

// comparison of the type is done using the name() method, because type_info
// are not shared between different linking units.
inline bool sameType(const std::type_info& t1, const std::type_info& t2) {
  return strcmp(t1.name(), t2.name()) == 0;
}

#else // SAFE_TYPE_COMPARISONS

inline bool sameType(const std::type_info& t1, const std::type_info& t2) {
  return t1 == t2;
}

#endif // SAFE_TYPE_COMPARISONS


// defined in src/base/essentia.cpp
std::string nameOfType(const std::type_info& type);

/**
 * Subclasses of this interface have the ability to check their type against
 * another one.
 */
class TypeProxy {
 protected:
  std::string _name;

 public:
  TypeProxy() {}
  TypeProxy(const std::string& name) : _name(name) {}

  virtual ~TypeProxy() {}

  const std::string& name() const { return _name; }
  void setName(const std::string& name) { _name = name; }

  inline void checkType(const std::type_info& received,
                        const std::type_info& expected) const {
    if (!sameType(received, expected)) {
      std::ostringstream msg;
      msg << "Error when checking types. Expected: " << nameOfType(expected)
          << ", received: " << nameOfType(received);
      throw EssentiaException(msg);
    }
  }

  template <typename Type>
  void checkType() const {
    checkType(typeid(Type), typeInfo());
  }

  void checkSameTypeAs(const TypeProxy& obj) const {
    checkType(obj.typeInfo(), typeInfo());
  }

  void checkVectorSameTypeAs(const TypeProxy& obj) const {
    checkType(obj.vectorTypeInfo(), typeInfo());
  }

  virtual const std::type_info& typeInfo() const = 0;
  virtual const std::type_info& vectorTypeInfo() const = 0;
};

/**
 * Commodity function to return the name of the type used in a TypeProxy.
 */
inline std::string nameOfType(const TypeProxy& tproxy) {
  return nameOfType(tproxy.typeInfo());
}

/**
 * Commodity function to compare two TypeProxy using their respective type.
 */
inline bool sameType(const TypeProxy& lhs, const TypeProxy& rhs) {
  return sameType(lhs.typeInfo(), rhs.typeInfo());
}

/**
 * Use this macro in classes that derive from TypeProxy to automatically
 * make them type-aware
 */
#define USE_TYPE_INFO(TokenType)                           \
  virtual const std::type_info& typeInfo() const {         \
    return typeid(TokenType);                              \
  }                                                        \
  virtual const std::type_info& vectorTypeInfo() const {   \
    return typeid(std::vector<TokenType>);                 \
  }


/**
 * typedef used for identifying sinks for a given source.
 */
typedef int ReaderID;

/**
 * Type used to represent a mono audio sample.
 */
typedef Real AudioSample;

template <typename T>
class Tuple2 {
 public:
  T first;
  T second;

  const T& left() const { return first; }
  const T& right() const { return second; }
  const T& x() const { return first; }
  const T& y() const { return second; }

  T& left() { return first; }
  T& right() { return second; }
  T& x() { return first; }
  T& y() { return second; }
};

/**
 * Type used to represent a stereo sample.
 */
typedef Tuple2<Real> StereoSample;



namespace streaming {

/**
 * This class is used to retrieve information about a buffer, such as its size,
 * phantom size, etc...
 * It is also used to pass this information to a buffer so it can resize itself.
 */
class BufferInfo {
 public:
  int size;
  int maxContiguousElements;

  BufferInfo(int size = 0, int contiguous = 0) :
    size(size), maxContiguousElements(contiguous) {}
};

namespace BufferUsage {

/**
 * Usage types for buffer which serve as preset sizes. The user can then
 * only specify for which kind of processing he will use this buffer and doesn't
 * need to know the specifics of the buffer implementation.
 */
enum BufferUsageType {
  forSingleFrames,
  forMultipleFrames,
  forAudioStream,
  forLargeAudioStream
};

} // namespace BufferUsage

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TYPES_H
