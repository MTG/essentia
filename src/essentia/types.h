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
 * Vector type to be used in Essentia.
 * Can either own its memory and act as a std::vector or be
 * be a const view on some other data it doesn't own.
 */
template <typename T>
class Vector {
 protected:
  bool _ownsMemory;
  // inefficient implementation for now, act as a stupid proxy
  std::vector<T> _buf;
  T* _data;
  Vector<T>* _source; // Vector containing the data if this is a view
  size_t _size;

 public:
  Vector() : _ownsMemory(true) {
    E_WARNING("Creating empty vector - owns = true [@" << this << "]");
  }

  Vector(uint size, T value = T()) : _ownsMemory(true), _buf(size, value) {
    E_WARNING("Creating vector, size = " << size << " - owns = true [@" << this << "]");
  }

  Vector(T* tab, size_t size) : _ownsMemory(false) {
    E_WARNING("Creating vector from fixed data, size = " << size << " - owns = false");
    setData(tab, size);
  }

  Vector(const Vector<T>& v/*, bool copy = false) : _ownsMemory(copy)*/) {
    bool copy = v.ownsMemory();
    _ownsMemory = copy;
    E_WARNING("Creating vector from other vector, size = " << v.size() << " - make copy = " << copy << " [this=@" << this << "] [other=@" << &v << "]");
    if (copy) {
      _buf = v.toStdVector();
    }
    else {
      // FIXME: should save a pointer to v, not v.data(),
      //        in case v does a v.resize(), to not invalidate the pointer
      //setData(v.data(), v.size());
      const Vector<T>* src = v.source();
      if (src) {
        setSource(*src);
      }
      else {
	setData(v.data(), v.size());
      }
    }
  }

    // FIXME: implement me
    static Vector* copy(const Vector<T>& v);
    static Vector* ref(const Vector<T>& v);

  ~Vector() {
    E_WARNING("deleting vector @" << this);
  }

  typedef const T* const_iterator;
  typedef T* iterator;

  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }
  iterator begin() { return data(); } // FIXME: only if ownsMemory
  iterator end() { return data() + size(); } // FIXME: only if ownsMemory

  inline bool ownsMemory() const { return _ownsMemory; }
  inline bool hasSource() const { return _source != 0; }
  inline const Vector<T>* source() const { return _source; }
  

  inline const T* data() const {
      E_WARNING("Buffer->data() const access, ownsMemory = " << _ownsMemory);
      if (_ownsMemory) return &_buf[0]; // FIXME: this segfaults if size==0
      else if (_source) return _source->data();
      else return this->_data;
  }

  inline T* data() {
      E_WARNING("Buffer->data() non-const access, ownsMemory = " << _ownsMemory);
      if (_ownsMemory) return &_buf[0]; // FIXME: this segfaults if size==0
      else if (_source) return _source->data();
      else return this->_data;
  }

  inline size_t size() const {
      if (_ownsMemory) return _buf.size();
      else if (_source) return _source->size();
      else return this->_size;
  }

  inline bool empty() const {
      return size() == 0;
  }

#define DECLARE_IF_OWN(returntype, f)                                                \
  returntype f() {                                                                   \
    E_WARNING("Vector::" #f "(), ownsMemory = " << _ownsMemory);                     \
    if (!_ownsMemory) {                                                              \
      throw EssentiaException("Cannot " #f " a Vector that doesn't own its memory"); \
    }                                                                                \
                                                                                     \
    return _buf.f();                                                                 \
  }

#define DECLARE_IF_OWN_1ARG(returntype, f, arg1type)                                 \
  returntype f(arg1type val) {                                                       \
    E_WARNING("Vector::" #f "(" #arg1type " val), val = " << val << ", ownsMemory = " << _ownsMemory); \
    if (!_ownsMemory) {                                                              \
      throw EssentiaException("Cannot " #f " a Vector that doesn't own its memory"); \
    }                                                                                \
                                                                                     \
    return _buf.f(val);                                                              \
  }

  DECLARE_IF_OWN(void, clear);
  DECLARE_IF_OWN(T&, front);

  DECLARE_IF_OWN_1ARG(void, resize, size_t);
  DECLARE_IF_OWN_1ARG(void, reserve, size_t);
  DECLARE_IF_OWN_1ARG(void, push_back, const T&);

    inline const T& operator[](int i) const {
        E_WARNING("Vector::op[] const, i = " << i);
        return data()[i];
    }

    inline T& operator[](int i) {
        E_WARNING("Vector::op[], i = " << i);
        return data()[i];
    }

    /**
     * Sets the data source to be a fixed region of memory.
     */
    void setData(const T* data, size_t size) {
        E_DEBUG(EMemory, "Vector::setData(" << data << ", " << size << ")");
        _ownsMemory = false;
        T* d = const_cast<T*>(data); // FIXME: hack
        this->_data = d;
        this->_size = size;
        this->_source = 0;
    }

    /**
     * Sets the data source to be another vector. Resizing the source vector
     * does not invalidate pointers.
     */
    void setSource(const Vector<T>& s) {
        E_DEBUG(EMemory, "Vector::setSource(" << s << ") [@" << &s << "]");
        _ownsMemory = false;
        Vector<T>* cs = &const_cast<Vector<T>&>(s); // FIXME: hack
        this->_data = 0;
        this->_size = 0;
        this->_source = cs;
    }

    std::vector<T> toStdVector() const {
        return std::vector<T>(begin(), end());
    }

};

/**
 * Output an essentia::Vector into an output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& v) {
  out << '['; if (!v.empty()) {
    out << *v.begin(); typename Vector<T>::const_iterator it = v.begin();
    for (++it; it != v.end(); ++it) out << ", " << *it;
  }
  return out << ']';
}


// TODO: also define Matrix type, this will allow to phase out TNT smoothly


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
    return typeid(Vector<TokenType>);                      \
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
