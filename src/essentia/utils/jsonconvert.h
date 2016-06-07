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


#ifndef JSON_CONVERT_H
#define JSON_CONVERT_H

#include <string>
#include "yamlast.h"

namespace essentia {

class JsonConvert {
 private:
  size_t _pos;
  std::string _str;
  std::string _result;
  size_t _size; 

 public:
  JsonConvert(const std::string& s) : _pos(0), _str(s), _result(""), _size(s.size()) {}
  virtual ~JsonConvert() {}
  std::string convert();


 protected:
  void skipSpaces();
  int countBackSlashes();
  std::string parseStringValue();
  std::string parseNumValue();
  std::string parseListValue();
  std::string parseDictKeyAndValue(const int& level);
 
 public: 
  std::string parseDict(const int& level=0);
};


class JsonException : public YamlException {
 public: 
  JsonException(const std::string& msg) : YamlException(msg) {}
};

} // namespace essentia

#endif // JSON_CONVERT_H