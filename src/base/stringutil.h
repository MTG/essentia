/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_STRINGUTIL_H
#define ESSENTIA_STRINGUTIL_H

#include <vector>
#include <string>
#include <sstream>

namespace essentia {

/**
 * Return a new string with the characters of str in lower-case.
 */
std::string toLower(const std::string& str);

/**
 * Return a new string with the characters of str in upper-case.
 */
std::string toUpper(const std::string& str);

/**
 * Return a string split whenever there is a char contained in the given
 * @c delimiters argument.
 */
std::vector<std::string> tokenize(const std::string& str, const std::string& delimiters, bool trimEmpty = false);

/**
 * Strip the given string of its leading and trailing whitespace characters.
 */
std::string strip(const std::string& str);


/**
 * The class Stringifier works a bit like the std::ostringstream but has better semantics.
 */
class Stringifier {
 protected:
  std::ostringstream oss;

 public:
  template <typename T>
  Stringifier& operator<<(const T& msg) { oss << msg; return *this; }

  std::string str() const { return oss.str(); }
};


std::string pad(int n, int size, char paddingChar=' ', bool leftPadded=false);
std::string pad(const std::string& str, int size, char paddingChar=' ', bool leftPadded=false);


} // namespace essentia

#endif // ESSENTIA_STRINGUTIL_H
