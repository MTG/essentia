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

#include "stringutil.h"
#include "essentiamath.h"
using namespace std;

namespace essentia {


string toLower(const string& str) {
  string result(str);
  for (int i=0; i<int(result.size()); ++i) {
    result[i] = std::tolower(str[i]);
  }
  return result;
}


string toUpper(const string& str) {
  string result(str);
  for (int i=0; i<int(result.size()); ++i) {
    result[i] = std::toupper(str[i]);
  }
  return result;
}


string strip(const string& str) {
  static string whitespace = " \t\n";
  size_t pos = str.find_first_not_of(whitespace);
  if (pos == string::npos) return string();

  size_t epos = str.find_last_not_of(whitespace);

  return str.substr(pos, epos-pos+1);
}


template <class Container>
void tokenize(const std::string& str, Container& tokens,
              const std::string& delimiters,
              const bool trimEmpty = false) {
  if (str.empty()) return;
  std::string::size_type pos, lastPos = 0;
  while (true) {
    pos = str.find_first_of(delimiters, lastPos);
    if (pos == std::string::npos) {
      pos = str.length();

      if (pos != lastPos || !trimEmpty) {
        tokens.push_back(typename Container::value_type(str.data() + lastPos,
                                                        pos - lastPos));
      }
      break;
    }

    if (pos != lastPos || !trimEmpty) {
      tokens.push_back(typename Container::value_type(str.data() + lastPos,
                                                      pos - lastPos));
    }

    lastPos = pos + 1;
  }
}



  vector<string> tokenize(const string& str, const string& delimiters, bool trimEmpty) {
  vector<string> result;
  tokenize(str, result, delimiters, trimEmpty);
  return result;
}


string pad(int n, int size, char paddingChar, bool leftPadded) {
  //if (n == 0) n = 1;
  ostringstream result;
  if (leftPadded) result << string(max<int>(0, size - ilog10(n) - 1), paddingChar) << n;
  else            result << n << string(max<int>(0, size - ilog10(n) - 1), paddingChar);
  return result.str();
}

string pad(const string& str, int size, char paddingChar, bool leftPadded) {
  if (leftPadded) return string(max<int>(0, size - str.size()), paddingChar) + str;
  else            return str + string(max<int>(0, size - str.size()), paddingChar);
}


} // namespace essentia
