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

#ifndef YAML_AST_H
#define YAML_AST_H

#include <string>
#include <map>
#include <vector>
#include <exception>
#include <sstream>
#include <yaml.h>

namespace essentia {

class YamlException : public std::exception {

 public:
  YamlException(const char* msg) : exception(), _msg(msg) {}
  YamlException(const std::string& msg) : exception(), _msg(msg) {}
  YamlException(const std::ostringstream& msg) : exception(), _msg(msg.str()) {}

  template <typename T, typename U>
  YamlException(const T& a, const U& b) : exception() {
    std::ostringstream oss; oss << a << b; _msg = oss.str();
  }

  template <typename T, typename U, typename V>
  YamlException(const T& a, const U& b, const V& c) : exception() {
    std::ostringstream oss; oss << a << b << c; _msg = oss.str();
  }

  virtual ~YamlException() throw() {}
  virtual const char* what() const throw() { return _msg.c_str(); }

 protected:
  std::string _msg;
};



class YamlNode {
 public:
  virtual ~YamlNode() = 0;
};


class YamlScalarNode : public YamlNode {
 public:
  enum YamlScalarType {
    STRING,
    FLOAT
  };

 private:
  YamlScalarType _tp;
  std::string _strDS;
  float _floatDS;

 public:
  YamlScalarNode(const float& f) : _tp(FLOAT), _floatDS(f) {}
  YamlScalarNode(const std::string& s) : _tp(STRING), _strDS(s) {}
  const YamlScalarType& getType() const { return _tp; }
  virtual ~YamlScalarNode() {}
  const std::string& toString() const {
    if (_tp != STRING) throw YamlException("YamlScalarNode is not a string");
    return _strDS;
  }
  const float& toFloat() const {
    if (_tp != FLOAT) throw YamlException("YamlScalarNode is not a float");
    return _floatDS;
  }
};


class YamlSequenceNode : public YamlNode {
 public:
  const std::vector<YamlNode*>& getData() const { return _data; }
  void add(YamlNode* n) { _data.push_back(n); }
  virtual ~YamlSequenceNode();
  const int size() const { return int(_data.size()); }
  const bool empty() const { return _data.empty(); }

 private:
  std::vector<YamlNode*> _data;
};


class YamlMappingNode : public YamlNode {
 public:
  const std::map<std::string, YamlNode*>& getData() const { return _data; }
  void add(const std::string& key, YamlNode* value) { _data[key] = value; }
  virtual ~YamlMappingNode();
  const int size() const { return int(_data.size()); }

 private:
  std::map<std::string, YamlNode*> _data;
};

YamlNode* parseYaml(FILE*);

} // namespace essentia

#endif // YAML_AST_H
