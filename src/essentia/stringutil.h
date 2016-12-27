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
